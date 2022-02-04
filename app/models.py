import torch
from torch.nn import CrossEntropyLoss, Module, Linear, GELU, Conv2d
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.loss import _WeightedLoss
from torchvision import models

resnet18 = models.resnet18(pretrained=False)


class InfoNCELoss(CrossEntropyLoss):
    # Oord et al 2019, Representation Learning with Contrastive Predictive Coding
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input_1, input_2, target):
        if target.dim() < 2:
            target.unsqueeze_(1)

        label_mask = torch.eye(target.shape[0]).to(target.device)
        # equivalent to -(F.log_softmax(input_x, dim=1) * label_mask).sum(1))
        return super().forward(input_2, label_mask)


def _verbose_snnl(score, target):
    """verbose implementation of soft nearest neighbour loss."""
    label_mask = torch.eq(target, target.T).type(torch.float32)

    negexp = torch.exp(score)
    negexp_zero_diag = negexp * (1 - torch.eye(target.shape[0]).to(target.device))

    return torch.log(
        (negexp_zero_diag * label_mask).sum(dim=1) / negexp_zero_diag.sum(dim=1) + torch.finfo(torch.float32).eps
    )


class SoftNearestNeighborsLoss(_WeightedLoss):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, input_1, input_2, target):
        if target.dim() < 2:
            target.unsqueeze_(1)

        label_mask = torch.eq(target, target.T).float()
        diagonal_mask = torch.diag(torch.stack([torch.tensor(-float("inf"))] * target.shape[0])).to(target.device)
        at_least_two_positives_mask = (label_mask.sum(dim=1) > 1.).unsqueeze(1).float()
        label_mask *= at_least_two_positives_mask

        # as of Frosst et al 2019
        score = -1 * torch.cdist(input_1, input_1, p=2).square() / self.temperature
        score += diagonal_mask

        loss = torch.log(
            (F.softmax(score, dim=1) * label_mask).sum(dim=1) + torch.finfo().eps
        )

        # assert torch.allclose(loss, _verbose_snnl(score, target))
        return (-1 * loss).mean()


class Network(Module):
    def __init__(self, cnn_block: Module, n_class: int = None) -> None:
        super().__init__()
        self.n_class = n_class

        self.layer_1 = Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_block = cnn_block
        self.layer_2 = GELU()
        self.layer_3 = Linear(1000, 64)
        self.proj_W = Parameter(torch.zeros(64, 64))
        torch.nn.init.xavier_normal_(self.proj_W)

        if self.n_class:
            self.logits_layer = Linear(64, self.n_class)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.cnn_block(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        if self.n_class:
            output_2 = self.logits_layer(x)
        else:
            # Oord et al 2019 uses log-bilinear as function f that approximates Mutual Information
            # project learned embeddings onto unit sphere
            norm_x = F.normalize(x)
            output_2 = norm_x @ self.proj_W @ norm_x.T

        return [x, output_2]
