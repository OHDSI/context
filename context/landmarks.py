import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy
import torch
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from context.model import create_embedding


@dataclass
class LandmarkDistances:
    rows: np.ndarray
    hops: np.ndarray
    components: int

    @cached_property
    def distance_profiles(self):
        order = np.lexsort(self.hops)
        starts = np.zeros(len(order), dtype=bool)
        starts[0] = True
        for row in self.hops:
            sorted_hops = row[order]
            starts[1:] |= sorted_hops[1:] != sorted_hops[:-1]
        positions = np.flatnonzero(starts)
        representatives = order[positions]
        counts = np.diff(np.append(positions, len(order)))
        group = np.empty(len(order), dtype=np.intp)
        group[order] = np.cumsum(starts) - 1
        return representatives, group, counts


def landmark_distances(dataset, root, count, seed):
    n = dataset.num_nodes
    if n == 0 or count < 2:
        raise ValueError("Landmark fitting requires a nonempty graph and a landmark budget of at least two")
    adjacency = dataset.adjacency(False)
    components, membership = connected_components(adjacency, directed=False)
    if components > 1:
        incoming = np.bincount(dataset.edges_list[:, 1].numpy(), minlength=n)
        degree = np.diff(adjacency.indptr)
        order = np.lexsort((np.arange(n), -degree, incoming != 0, membership))
        first = np.r_[True, np.diff(membership[order]) != 0]
        representatives = order[first]
        if root is not None:
            representatives[membership[root]] = root
        bridges = csr_matrix((np.ones(components, dtype=bool), (representatives, np.zeros(components, dtype=int))),
                             shape=(n, 1))
        adjacency = bmat([[adjacency, bridges], [bridges.T, None]], format="csr")
    priority = np.random.default_rng(seed).permutation(n)
    pivot = int(priority[0]) if root is None else root
    rows = np.empty(min(count, n), dtype=np.intp)
    hops = np.empty((len(rows), n), dtype=np.int32)
    nearest = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    for i in range(len(rows)):
        rows[i] = pivot
        hops[i] = dijkstra(adjacency, directed=True, indices=pivot, unweighted=True)[:n]
        np.minimum(nearest, hops[i], out=nearest)
        pivot = int(priority[np.argmax(nearest[priority])])
    return LandmarkDistances(rows, hops, int(components))


def _spectral_projection(distances, dimensions, curvature):
    if curvature is None:
        squared = np.square(distances)
        center = squared.mean(axis=0)
        gram = -.5 * (squared - center - center[:, None] + center.mean())
    else:
        gram = np.cosh(math.sqrt(curvature) * distances)
        center = None
    values, vectors = np.linalg.eigh(gram)
    tolerance = len(values) * np.finfo(float).eps * np.max(np.abs(values))
    if curvature is None:
        axes = np.flatnonzero(values > tolerance)[::-1][:dimensions]
        projection = -.5 * vectors[:, axes] / np.sqrt(values[axes])
    else:
        axes = np.flatnonzero(values < -tolerance)[:dimensions]
        projection = -vectors[:, axes] / np.sqrt(-values[axes])
    return center, projection


def _spatial_coordinates(hops, edge_length, center, projection, curvature):
    distances = np.asarray(hops, dtype=np.float64) * edge_length
    if curvature is None:
        features = np.square(distances) - center
    else:
        features = np.cosh(math.sqrt(curvature) * distances)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        coordinates = features @ projection
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite landmark projection. Reduce landmark_edge_length.")
    return coordinates


def _poincare_coordinates(spatial, root, curvature):
    time = np.sqrt(1 + np.einsum("ij,ij->i", spatial, spatial))
    if root is not None:
        root_time = math.sqrt(1 + np.einsum("i,i->", root, root))
        spatial = spatial + (np.einsum("ij,j->i", spatial, root) / (1 + root_time) - time)[:, None] * root
        time = np.sqrt(1 + np.einsum("ij,ij->i", spatial, spatial))
    return spatial / ((1 + time)[:, None] * math.sqrt(curvature))


def fit_coordinates(basis, args, root):
    n = basis.hops.shape[1]
    anchor = None if args.geometry == "euclidean" and args.euclidean_root == "trainable" else root
    with torch.random.fork_rng(devices=[]):
        model = create_embedding(n, args.embedding_dim, args.geometry, args.curvature, anchor, args.euclidean_distance_scale)
    curvature = model.curvature.item() if model.curvature is not None else None
    max_hops = int(basis.hops.max())
    length = args.landmark_edge_length
    if length is None:
        length = min(1., 6 / (math.sqrt(curvature) * max(1, max_hops))) if curvature else 1.
    if not math.isfinite(length) or length <= 0:
        raise ValueError("landmark_edge_length must be finite and positive")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"), torch.no_grad():
            center, projection = _spectral_projection(basis.hops[:, basis.rows].astype(float) * length,
                                                       args.embedding_dim, curvature)
            root_coordinates = None
            if anchor is not None:
                root_coordinates = _spatial_coordinates(basis.hops[:, anchor], length, center, projection, curvature)
            representatives, group, counts = basis.distance_profiles
            unique_coordinates = np.empty((len(representatives), projection.shape[1]), dtype=np.float32)
            for start in range(0, len(representatives), 8192):
                block = slice(start, start + 8192)
                points = _spatial_coordinates(basis.hops[:, representatives[block]].T, length, center, projection, curvature)
                if curvature is not None:
                    points = _poincare_coordinates(points, root_coordinates, curvature)
                else:
                    if root_coordinates is not None:
                        points -= root_coordinates
                    points /= model.distance_scale
                coordinates = points.astype(np.float32)
                if not np.isfinite(coordinates).all() or (curvature and
                        np.any(curvature * np.square(coordinates.astype(float)).sum(axis=1) >= 1)):
                    raise ValueError("Coordinates exceed export precision. Reduce landmark_edge_length.")
                coordinates[coordinates == 0] = 0.0
                unique_coordinates[block] = coordinates
            if anchor is not None:
                unique_coordinates[group[anchor]] = 0.
            model.coordinates().zero_()
            for start in range(0, n, 32768):
                block = slice(start, start + 32768)
                model.coordinates()[block, :projection.shape[1]].copy_(torch.from_numpy(unique_coordinates[group[block]]))
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Unstable landmark spectrum. Reduce landmark_edge_length.") from error
    if projection.shape[1]:
        row_type = np.dtype((np.void, unique_coordinates.itemsize * unique_coordinates.shape[1]))
        _, row_groups = np.unique(unique_coordinates.view(row_type).ravel(), return_inverse=True)
        multiplicity = np.zeros(row_groups.max() + 1, dtype=np.int64)
        np.add.at(multiplicity, row_groups, counts)
    else:
        multiplicity = np.array([n])
    details = {"distinct_distance_profiles": len(representatives), "distinct_coordinate_rows": len(multiplicity),
               "nodes_sharing_coordinates": int(multiplicity[multiplicity > 1].sum()),
               "largest_tied_group": int(multiplicity.max()), "numpy_version": np.__version__,
               "scipy_version": scipy.__version__, "algorithm": "landmark_mds" if curvature is None else "l_hydra",
               "landmark_rows": basis.rows.tolist(), "effective_dimensions": projection.shape[1],
               "edge_length": length, "maximum_landmark_hops": max_hops, "training_components": basis.components,
               "synthetic_hub_links": basis.components if basis.components > 1 else 0,
               "graph_metric": "undirected shortest path on training edges", "compute_device": "cpu"}
    return model.eval(), details
