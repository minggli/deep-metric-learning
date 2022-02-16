import torch

from torch import nn
import ngtpy


def initiate_indexer(path: str, embedding_dim: int) -> ngtpy.Index:
    ngtpy.create(path, embedding_dim)
    return ngtpy.Index(path)


def embed(raw_data: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        vectors, _ = model(raw_data)
    
    return vectors.detach().cpu().numpy()


def fit_indexer(raw_data: torch.Tensor, index: ngtpy.Index, model: nn.Module) -> None:
    vectors = embed(raw_data, model)
    index.batch_insert(vectors)


def query_indexer(raw_query: torch.Tensor, index: ngtpy.Index, model: nn.Module, top_k: int = 10) -> list:
    query_vector = embed(raw_query, model)
    return index.search(query_vector, size=top_k)
