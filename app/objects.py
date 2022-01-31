import torch
from torchvision import transforms as T

from app.utils import get_torch_device


def assign_device(tensor: torch.Tensor, device=get_torch_device()):
    return tensor.to(device)


# pylint: disable=too-few-public-methods
class BaseTransform(T.Compose):
    pass


class ImageTransform(BaseTransform):
    def __init__(self, transforms):
        _ = transforms
        super().__init__(
            [
                T.ToTensor(),
                T.Lambda(assign_device),
            ]
        )


class TextTransform(BaseTransform):
    pass
