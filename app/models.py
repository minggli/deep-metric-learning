from torch import nn
from torchvision import models
# from torch.nn.modules.loss import _WeightedLoss

resnet18 = models.resnet18(pretrained=False)

# class InfoNCELoss(_WeightedLoss):
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super().__init__(weight, size_average, reduce, reduction)
    
#     def forward(pos_pair, neg_pair)


class SiameseNetwork(nn.Module):
    def __init__(self, pre_trained_model: nn.Module, n_class: int = None) -> None:
        super().__init__()
        if n_class:
            self.n_class = n_class
        else:
            self.n_class = 1

        self.layer_1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.pre_trained_model = pre_trained_model
        self.layer_2 = nn.GELU()
        self.logits = nn.Linear(1000, self.n_class)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.pre_trained_model(x)
        x = self.layer_2(x)
        return self.logits(x)
