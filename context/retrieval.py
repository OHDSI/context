import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


def edge_index(edges, num_nodes):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("Edge index outside the coordinate table")
    result = csr_matrix((np.ones(len(edges), dtype=bool), edges.T), shape=(num_nodes, num_nodes))
    result.sort_indices()
    return result


def query_rows(adjacency, limit, seed):
    eligible = np.flatnonzero(np.diff(adjacency.indptr))
    if limit < 0:
        raise ValueError("Query limit must be nonnegative")
    if limit and limit < len(eligible):
        return np.sort(np.random.default_rng(seed).choice(eligible, limit, replace=False))
    return eligible


def _neighbors(adjacency, source):
    start, stop = adjacency.indptr[source:source + 2]
    return adjacency.indices[start:stop]


def _score_counts(scores, thresholds):
    if len(thresholds) == 1:
        return np.count_nonzero(scores < thresholds[0]), np.count_nonzero(scores <= thresholds[0])
    if len(thresholds) <= 12:
        less = np.fromiter((np.count_nonzero(scores < value) for value in thresholds), dtype=np.int64)
        through = np.fromiter((np.count_nonzero(scores <= value) for value in thresholds), dtype=np.int64)
        return less, through
    insertion_left = np.searchsorted(thresholds, scores, side="left")
    insertion_right = np.searchsorted(thresholds, scores, side="right")
    less = np.cumsum(np.bincount(insertion_right, minlength=len(thresholds) + 1))[:-1]
    through = np.cumsum(np.bincount(insertion_left, minlength=len(thresholds) + 1))[:-1]
    return less, through


def rank_neighbors(points, targets, queries, curvature=None, known=None, batch_size=32, candidate_block=32768):
    points = np.asarray(points, dtype=np.float64, order="C")
    queries = np.asarray(queries, dtype=np.int64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Expected a finite matrix of coordinates")
    if not len(queries) or queries.min() < 0 or queries.max() >= len(points):
        raise ValueError("Eligible query rows are required")
    if batch_size < 1 or candidate_block < 1:
        raise ValueError("Ranking block sizes must be positive")
    margin = None
    if curvature is not None:
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError("Curvature must be finite and positive")
        margin = 1 - curvature * np.einsum("ij,ij->i", points, points)
        if (margin <= 0).any():
            raise ValueError("Coordinates lie outside the open Poincaré ball")
    edge_rank_sum = macro_rank_sum = ap_sum = 0.0
    positives_count = 0
    for first in range(0, len(queries), batch_size):
        sources = queries[first:first + batch_size]
        groups = []
        for source in sources:
            relevant = _neighbors(targets, source)
            if not len(relevant) or np.any(relevant == source):
                raise ValueError("Each query requires non-self positive targets")
            excluded = np.array([source]) if known is None else np.union1d(
                np.setdiff1d(_neighbors(known, source), relevant, assume_unique=True), [source])
            scores = cdist(points[source:source + 1], points[relevant], metric="sqeuclidean")[0]
            if margin is not None:
                scores /= margin[relevant]
            thresholds, counts = np.unique(scores, return_counts=True)
            groups.append((thresholds, counts, excluded, np.zeros_like(counts), np.zeros_like(counts)))
        for start in range(0, len(points), candidate_block):
            stop = min(start + candidate_block, len(points))
            scores = cdist(points[sources], points[start:stop], metric="sqeuclidean")
            if margin is not None:
                scores /= margin[None, start:stop]
            for row, (thresholds, _, excluded, less, through) in zip(scores, groups):
                left, right = excluded.searchsorted([start, stop])
                row[excluded[left:right] - start] = np.inf
                below, at_or_below = _score_counts(row, thresholds)
                less += below
                through += at_or_below
        for _, counts, _, less, through in groups:
            positive_through = np.cumsum(counts)
            positives = positive_through[-1]
            midpoint = (less + through + 1) / 2
            filtered_rank = 1 + (less + through - 2 * positive_through + counts) / 2
            edge_rank_sum += counts @ filtered_rank
            macro_rank_sum += (counts @ midpoint) / positives
            ap_sum += (counts @ (positive_through / through)) / positives
            positives_count += positives
    return {"mean_rank": float(edge_rank_sum / positives_count), "macro_average_precision": float(ap_sum / len(queries)),
            "macro_mean_positive_midrank": float(macro_rank_sum / len(queries)), "queries": len(queries),
            "eligible_queries": int(np.count_nonzero(np.diff(targets.indptr))), "candidates": len(points),
            "positive_edges_evaluated": int(positives_count)}


def evaluate(points, geometry, targets, known, queries, curvature=1.0, distance_scale=1.0):
    if geometry not in ("euclidean", "hyperbolic") or not np.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("Expected a supported geometry and a positive distance scale")
    return rank_neighbors(points, edge_index(targets, len(points)), queries,
                          curvature if geometry == "hyperbolic" else None, edge_index(known, len(points)))
