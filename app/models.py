import torch
from torch import nn
from torchvision import models

resnet18 = models.resnet18(pretrained=False)

class InfoNCELoss(nn.CrossEntropyLoss):
    # Oord et al 2019
    # nn.CrossEntroyLoss includes log-softmax operation before cross entropy

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.projection_W = nn.Parameter(torch.zeros(64, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.projection_W)
    
    def forward(self, x, y):
        # project learned embeddings onto unit sphere
        norm_x = nn.functional.normalize(x)
        # original paper uses log-bilinear as f
        score = norm_x @ self.projection_W @ norm_x.T
        class_indice = torch.arange(norm_x.shape[0]).to(norm_x.device)

        return super().forward(score, class_indice)


class SiameseNetwork(nn.Module):
    def __init__(self, cnn_block: nn.Module, n_class: int = None) -> None:
        super().__init__()

        self.layer_1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_block = cnn_block
        self.layer_2 = nn.GELU()
        self.output_layer = nn.Linear(1000, 64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.cnn_block(x)
        x = self.layer_2(x)
        return self.output_layer(x)
