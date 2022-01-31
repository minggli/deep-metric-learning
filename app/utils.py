from pathlib import Path
import torch

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_project_root() -> Path:
    return Path(__file__).parent.parent
