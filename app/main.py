from time import time

import torch
import yaml
from smart_open import open
from torch import nn
from torch.utils.data import DataLoader

from app.datasets import ExperimentDatasets, load_dataset
from app.ml_ops import test, train
from app.models import InfoNCELoss, Network, SoftNearestNeighborsLoss, resnet18
from app.objects import ImageTransform, TargetTransform
from app.utils import get_project_root, get_torch_device

PROJ_ROOT = get_project_root()
with open(PROJ_ROOT / "app/model_cfg.yaml", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)


if __name__ == "__main__":
    ds_train, ds_test = load_dataset(
        ExperimentDatasets.FASHION_MNIST, transformer=ImageTransform, target_transformer=TargetTransform
    )
    train_batch_iter, test_batch_iter = DataLoader(
        ds_train, batch_size=model_config["batch_size"], shuffle=True, num_workers=0
    ), DataLoader(ds_test, batch_size=model_config["batch_size"], shuffle=True, num_workers=0)

    model = Network(resnet18).to(get_torch_device())
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = SoftNearestNeighborsLoss().to(get_torch_device())
    loss = nn.DataParallel(loss)

    timestamp = int(time())

    for epoch in range(10):
        train(train_batch_iter, model, loss, optimizer)

    with open(f"gs://saved_models_minggli/model_{timestamp}.pt", "wb") as f:
        torch.save(model, f)
