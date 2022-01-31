from enum import Enum
from typing import cast

from torchvision.datasets import MNIST, FashionMNIST, ImageNet, VisionDataset
from app.objects import BaseTransform, TargetTransform
from app.utils import get_project_root

PROJ_ROOT = get_project_root()
DATA_ROOT = str(PROJ_ROOT / "downloads")


def load_dataset(
    dataset: "ExperimentDatasets",
    transformer: type[BaseTransform],
    target_transformer: type[TargetTransform],
    root: str = DATA_ROOT,
) -> tuple[VisionDataset, VisionDataset]:
    dataset_klass = dataset.value
    dataset_klass = cast(MNIST, dataset_klass)

    transformer_instance: BaseTransform = transformer([])
    target_transformer_instance: TargetTransform = target_transformer([])

    return (
        dataset_klass(root, train=True, download=True, transform=transformer_instance, target_transform=target_transformer_instance),
        dataset_klass(root, train=False, download=True, transform=transformer_instance, target_transform=target_transformer_instance),
    )


class CustomImageNet(ImageNet):
    def __init__(self, root: str, train: bool = True, download=True, **kwargs):
        if train:
            split = "train"
        else:
            split = "val"
        super().__init__(root, split=split, download=download, **kwargs)


class ExperimentDatasets(Enum):
    """a set of built-in datasets for this project."""

    MNIST = MNIST
    FASHION_MNIST = FashionMNIST
    IMAGENET = CustomImageNet
