import torch

from torch import nn
import ngtpy


def initiate_indexer(path: str, embedding_dim: int) -> ngtpy.Index:
    ngtpy.create(path, embedding_dim)
    return ngtpy.Index(path)


def fit_indexer(raw_data: torch.Tensor, index: ngtpy.Index, model: nn.Module) -> None:
    model.eval()
    with torch.no_grad():
        vectors, _ = model(raw_data)
    index.batch_insert(vectors)


def query_indexer(raw_query: torch.Tensor, index: ngtpy.Index, model: nn.Module, top_k: int = 10) -> list:
    model.eval()
    with torch.no_grad():
        query_vector, _ = model(raw_query)

    return index.search(query_vector, k=top_k)
