import math

import numpy as np
import torch

from context.model import pairwise_distance


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
                               geometry="hyperbolic", curvature=1.0):
    """Return filtered mean positive rank (edge-weighted) and MAP (query-weighted)."""
    if geometry not in ("euclidean", "hyperbolic"):
        raise ValueError(f"Unknown embedding geometry {geometry!r}")
    if num_nodes != len(embeddings) or num_nodes != dataset.num_nodes or batch_size < 1:
        raise ValueError("Graph and embedding sizes must match and batch size must be positive")
    c = curvature if geometry == "hyperbolic" else None
    dtype = torch.float32 if embeddings.device.type == "mps" else torch.float64
    points = embeddings.to(dtype=dtype)
    _check_points(points, c)
    adjacency = dataset.generate_adjacency_sets(directed)
    queries = [i for i, neighbors in enumerate(adjacency) if neighbors]
    rank_sum = ap_sum = 0.0
    positive_count = 0
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        rows = pairwise_distance(points[batch], points, c).cpu().numpy()
        for source, row in zip(batch, rows):
            neighbors = list(adjacency[source])
            row[source] = np.inf
            relevant = np.sort(row[neighbors])
            ordered = np.sort(row)
            left = ordered.searchsorted(relevant, "left") - relevant.searchsorted(relevant, "left")
            through = ordered.searchsorted(relevant, "right")
            positives_through = relevant.searchsorted(relevant, "right")
            right = through - positives_through
            rank_sum += (1 + (left + right) / 2).sum()
            ap_sum += (positives_through / through).mean()
            positive_count += len(neighbors)
    return float(rank_sum / positive_count), float(ap_sum / len(queries))


def evaluate_model(model, dataset, batch_size=64, directed=None):
    c = None if model.curvature is None else model.curvature.item()
    return evaluate_mean_rank_and_map(dataset, model.coordinates().detach(), dataset.num_nodes,
                                      batch_size, dataset.directed if directed is None else directed,
                                      model.geometry, c)
