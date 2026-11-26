import math

import torch

from context.model import pairwise_distance
from context.retrieval import query_rows, rank_neighbors


def _check_points(points, curvature):
    if points.ndim != 2 or not torch.isfinite(points).all():
        raise ValueError("Expected a finite matrix of embedding coordinates")
    if curvature is not None:
        if not math.isfinite(curvature) or curvature <= 0:
            raise ValueError("Curvature magnitude must be finite and positive")
        if (curvature * points.square().sum(-1) >= 1).any():
            raise ValueError("Coordinates lie outside the open Poincaré ball")


def hyperbolic_distance(u, v, c=1.0, eps=1e-5):
    """Return pairwise Poincaré distances for sectional curvature -c."""
    _check_points(u, c)
    _check_points(v, c)
    return pairwise_distance(u, v, c)


def fast_average_precision(y_true, y_score):
    scores, order = y_score.sort(descending=True)
    labels = y_true[order].bool()
    if not labels.any():
        return y_score.new_zeros(())
    ends = torch.cat((torch.where(scores[:-1] != scores[1:])[0] + 1, order.new_tensor([len(scores)])))
    counts = labels.cumsum(0)[ends - 1]
    return (counts.diff(prepend=counts.new_zeros(1)) * counts / ends).sum() / counts[-1]


@torch.no_grad()
def evaluate_mean_rank_and_map(dataset, embeddings, num_nodes, batch_size=128, directed=False,
                               geometry="hyperbolic", curvature=1.0, query_limit=0, query_seed=1):
    """Return filtered mean positive rank (edge-weighted) and MAP (query-weighted)."""
    if geometry not in ("euclidean", "hyperbolic"):
        raise ValueError(f"Unknown embedding geometry {geometry!r}")
    if num_nodes != len(embeddings) or num_nodes != dataset.num_nodes or batch_size < 1:
        raise ValueError("Graph and embedding sizes must match and batch size must be positive")
    adjacency = dataset.adjacency(directed)
    result = rank_neighbors(embeddings.detach().cpu().numpy(), adjacency, query_rows(adjacency, query_limit, query_seed),
                            curvature if geometry == "hyperbolic" else None, batch_size=min(32, batch_size))
    return result["mean_rank"], result["macro_average_precision"]


def evaluate_model(model, dataset, batch_size=64, directed=None, query_limit=0, query_seed=1):
    curvature = None if model.curvature is None else model.curvature.item()
    return evaluate_mean_rank_and_map(dataset, model.coordinates().detach(), dataset.num_nodes,
                                      batch_size, dataset.directed if directed is None else directed,
                                      model.geometry, curvature, query_limit, query_seed)
