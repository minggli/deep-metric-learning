import yaml

import torch
import torch.nn.functional as F
from torch.nn import GELU, Conv2d, CrossEntropyLoss, Linear, Module, Parameter
from torch.nn.modules.loss import _WeightedLoss

from app.utils import get_project_root


PROJ_ROOT = get_project_root()
with open(PROJ_ROOT / "app/model_cfg.yaml", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)
    embedding_dim = model_config["embedding_dim"]


class InfoNCELoss(CrossEntropyLoss):
    # Oord et al 2019, Representation Learning with Contrastive Predictive Coding
    # Identity one positive example against noise examples
    def __init__(self, *args, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, input_1, input_2, target):
        if target.dim() < 2:
            target.unsqueeze_(1)

        positive_mask = torch.eq(target, target.T)
        diag_mask = torch.eye(target.shape[0]).bool().to(target.device)
        off_diag_pos_mask = torch.logical_and(positive_mask, ~diag_mask)
        # ignore off-diagonal positive example by sending logits to negative infinity before softmax
        off_diag_pos_inf = torch.where(
            off_diag_pos_mask,
            torch.log(torch.tensor(0.0) + torch.finfo().eps).to(target.device),
            torch.tensor(0.0).to(target.device),
        )
        score = (input_2 + off_diag_pos_inf) / self.temperature

        label_mask = torch.eye(target.shape[0]).to(target.device)
        # equivalent to -(F.log_softmax(score, dim=1) * label_mask).sum(1).mean()
        return super().forward(score, label_mask)


class SoftNearestNeighborsLoss(_WeightedLoss):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, input_1, input_2, target):
        if target.dim() < 2:
            target.unsqueeze_(1)

        positive_mask = torch.eq(target, target.T).float()
        diagonal_inf = torch.diag(torch.stack([torch.tensor(-float("inf"))] * target.shape[0])).to(target.device)
        at_least_two_positives_mask = (positive_mask.sum(dim=1) > 1.0).unsqueeze(1).float()
        positive_mask *= at_least_two_positives_mask

        # as of Frosst et al 2019
        score = -1 * torch.cdist(input_1, input_1, p=2).square() / self.temperature
        score += diagonal_inf

        loss = torch.log((F.softmax(score, dim=1) * positive_mask).sum(dim=1) + torch.finfo().eps)
        return (-1 * loss).mean()


class Network(Module):
    def __init__(self, cnn_block: Module, n_class: int = None, embedding_dim: int = embedding_dim) -> None:
        super().__init__()
        self.n_class = n_class

        self.layer_1 = Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_block = cnn_block
        self.layer_2 = GELU()
        self.layer_3 = Linear(1000, embedding_dim)
        self.proj_W = Parameter(torch.zeros(embedding_dim, embedding_dim))

        if self.n_class:
            self.logits_layer = Linear(embedding_dim, self.n_class)

        torch.nn.init.xavier_normal_(self.proj_W)

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
