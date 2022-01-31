import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from app.datasets import ExperimentDatasets, load_dataset
from app.ml_ops import test, train
from app.models import SiameseNetwork, resnet18
from app.objects import ImageTransform
from app.utils import get_project_root, get_torch_device

PROJ_ROOT = get_project_root()

with open(PROJ_ROOT / "app/model_cfg.yaml", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)


if __name__ == "__main__":
    ds_train, ds_test = load_dataset(ExperimentDatasets.FASHION_MNIST, transformer=ImageTransform)
    train_batch_iter, test_batch_iter = DataLoader(
        ds_train, batch_size=model_config["batch_size"], shuffle=True, num_workers=4
    ), DataLoader(ds_test, batch_size=model_config["batch_size"], shuffle=True, num_workers=4)

    n_class = len(ds_train.classes)
    model = SiameseNetwork(resnet18, n_class=n_class).to(get_torch_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    for epoch in range(10):
        train(train_batch_iter, model, loss, optimizer)
        test(test_batch_iter, model, loss)
