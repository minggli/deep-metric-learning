from time import time

import torch
import yaml
from smart_open import open
from torch import nn
from torch.utils.data import DataLoader

from app.datasets import ExperimentDatasets, load_dataset
from app.ml_ops import test_single_batch, train, visualise_embedding
from app.models import InfoNCELoss, Network, SoftNearestNeighborsLoss, resnet18
from app.objects import ImageTransform, TargetTransform
from app.utils import get_project_root, get_torch_device

PROJ_ROOT = get_project_root()
with open(PROJ_ROOT / "app/model_cfg.yaml", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)


if __name__ == "__main__":
    dataset = getattr(ExperimentDatasets, model_config['dataset'])
    dataset_name = str(dataset.name)

    ds_train, ds_test = load_dataset(dataset, transformer=ImageTransform, target_transformer=TargetTransform)
    train_batch_iter, test_batch_iter = DataLoader(
        ds_train, batch_size=model_config["batch_size"], shuffle=True, num_workers=0
    ), DataLoader(ds_test, batch_size=model_config["batch_size"], shuffle=False, num_workers=0)

    n_class = ds_train.targets.unique().shape[0]
    model = Network(resnet18, n_class=None).to(get_torch_device())
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss: nn.Module = InfoNCELoss().to(get_torch_device())
    loss = nn.DataParallel(loss)
    loss_name = getattr(loss, "module", loss).__class__.__name__

    timestamp = int(time())
    x_test, y_test = next(iter(test_batch_iter))

    images: list = []
    visualise_embedding(loss_name, dataset_name, 0, images, x_test, y_test, model)
    for epoch in range(1, 10):
        train(train_batch_iter, model, loss, optimizer)
        visualise_embedding(loss_name, dataset_name, epoch, images, x_test, y_test, model)
        test_single_batch(x_test, y_test, model, loss)

    with open(f"gs://saved_models_minggli/images_{timestamp}.gif", "wb") as f:
        images[0].save(f, format="GIF", append_images=images[1:], save_all=True, duration=2000, loop=0)

    with open(f"gs://saved_models_minggli/model_{timestamp}.pt", "wb") as f:
        torch.save(model, f)
