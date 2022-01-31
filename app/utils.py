from pathlib import Path
import torch


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_project_root() -> Path:
    return Path(__file__).parent.parent
