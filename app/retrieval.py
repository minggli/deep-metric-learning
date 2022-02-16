from time import time

import torch
import yaml
from smart_open import open
from torch import nn
from torch.utils.data import DataLoader

from app.datasets import ExperimentDatasets, load_dataset
from app.ml_ops import test_single_batch
from app.utils import get_project_root, get_torch_device
from app.search import initiate_indexer, fit_indexer, query_indexer
from app.objects import ImageTransform, TargetTransform


if __name__ == "__main__":
    PROJ_ROOT = get_project_root()
    with open(PROJ_ROOT / "app/model_cfg.yaml", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    dataset = getattr(ExperimentDatasets, model_config["dataset"])
    dataset_name = str(dataset.name)
    retrieval_model_id = model_config["retrieval_model_id"]
    indexer_prefix = model_config["indexer_prefix"]
    embedding_dim = model_config["embedding_dim"]

    _, ds_test = load_dataset(dataset, transformer=ImageTransform, target_transformer=TargetTransform)
    test_batch_iter = DataLoader(ds_test, batch_size=model_config["batch_size"], shuffle=False, num_workers=0)
    x_test, y_test = next(iter(test_batch_iter))

    with open(f"gs://saved_models_minggli/model_{retrieval_model_id}.pt", "rb") as f:
        model = torch.load(f)

    indexer_bytes_path = f"{indexer_prefix}_{retrieval_model_id}".encode()
    indexer = initiate_indexer(indexer_bytes_path, embedding_dim)
