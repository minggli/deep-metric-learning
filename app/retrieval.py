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

from app.search import initiate_indexer, fit_indexer, query_indexer

PROJ_ROOT = get_project_root()
with open(PROJ_ROOT / "app/model_cfg.yaml", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)


dataset = getattr(ExperimentDatasets, model_config['dataset'])
dataset_name = str(dataset.name)

_, ds_test = load_dataset(dataset, transformer=ImageTransform, target_transformer=TargetTransform)
test_batch_iter = DataLoader(ds_test, batch_size=model_config["batch_size"], shuffle=False, num_workers=0)
x_test, y_test = next(iter(test_batch_iter))

with open(f"gs://saved_models_minggli/indexer_{timestamp}.pt", "wb") as f:
    initiate_indexer()