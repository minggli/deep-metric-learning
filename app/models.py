import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

resnet18 = models.resnet18(pretrained=False)


class InfoNCELoss(nn.CrossEntropyLoss):
    # Oord et al 2019, Representation Learning with Contrastive Predictive Coding
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, reduction='none', **kwargs)

    def forward(self, input_1, input_2, target):
        if target.dim() < 2:
            target.unsqueeze_(1)
        
        label_mask = torch.eye(target.shape[0]).to(target.device)
        # equivalent to -(F.log_softmax(input_x, dim=1) * label_mask).sum(1))
        loss = super(InfoNCELoss, self).forward(input_2, label_mask)

        return loss.mean()


class SoftNearestNeighborsLoss(_WeightedLoss):
    def __init__(self, *args, temperature: float = 1., **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, input_1, input_2, target):
        if target.dim() < 2:
            target.unsqueeze_(1)

        score = -1 * (torch.cdist(input_1, input_1, p=2).square() / self.temperature)
        label_mask = torch.eq(target, target.T).type(torch.float32)

        loss = torch.log(
            (F.softmax(score, dim=1) * label_mask).sum(dim=1) \
            + torch.finfo(torch.float32).eps
        )
        loss *= -1

        return loss.mean()


class Network(nn.Module):
    def __init__(self, cnn_block: nn.Module) -> None:
        super().__init__()

        self.layer_1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_block = cnn_block
        self.layer_2 = nn.GELU()
        self.layer_3 = nn.Linear(1000, 64)
        self.proj_W = nn.Parameter(torch.zeros(64, 64))
        torch.nn.init.xavier_normal_(self.proj_W)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.cnn_block(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        # project learned embeddings onto unit sphere
        norm_x = nn.functional.normalize(x)

        # Oord et al 2019 uses log-bilinear as function f that approximates Mutual Information
        return [norm_x, norm_x @ self.proj_W @ norm_x.T]

