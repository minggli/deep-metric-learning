from collections import namedtuple
from enum import Enum
from typing import cast

from torchvision.datasets import MNIST, FashionMNIST, VisionDataset

from app.utils import get_project_root

DATA_ROOT = str(get_project_root() / "downloads")


def load_dataset(dataset: "ExperimentDatasets", root: str = DATA_ROOT) -> tuple[VisionDataset, VisionDataset]:
    dataset_klass = dataset.value
    dataset_klass = cast(MNIST, dataset_klass)

    return (dataset_klass(root, train=True, download=True), dataset_klass(root, train=False, download=True))


class ExperimentDatasets(Enum):
    """a set of built-in datasets for this project."""

    MNIST = MNIST
    FASHION_MNIST = FashionMNIST
