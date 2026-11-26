import math

import numpy as np
import rustworkx as rx
import scipy
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, aslinearoperator, eigsh, spsolve_triangular
import torch

from context.model import create_embedding


METHODS = ("random", "adjacency_svd", "laplacian", "fastrp", "ancestor", "hope", "deepwalk", "node2vec")


def validate_settings(args):
    if args.geometry != "euclidean" or str(args.device) != "cpu":
        raise ValueError(f"{args.training_method} produces native Euclidean vectors on CPU. Use --geometry euclidean --device cpu.")
    if args.embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")
    method = args.training_method
    if method in ("ancestor", "fastrp") and args.diffusion_steps < 1:
        raise ValueError("diffusion_steps must be positive")
    if method == "fastrp" and not math.isfinite(args.degree_exponent):
        raise ValueError("degree_exponent must be finite")
    if method == "ancestor" and not (0 <= args.ancestor_mix <= 1 and 0 < args.ancestor_decay <= 1):
        raise ValueError("ancestor_mix must lie in [0,1] and ancestor_decay in (0,1]")
    if method == "hope" and (args.embedding_dim < 2 or not math.isfinite(args.katz_beta) or args.katz_beta <= 0):
        raise ValueError("HOPE requires at least two coordinates and a finite, positive katz_beta")
    if method in ("hope", "adjacency_svd") and args.svd_iterations < 0:
        raise ValueError("svd_iterations must be nonnegative")
    if method == "laplacian" and not (0 < args.spectral_tolerance < 1 and args.spectral_iterations > 0):
        raise ValueError("Laplacian fitting requires a positive iteration budget and tolerance in (0,1)")
    if method in ("deepwalk", "node2vec"):
        from context.walks import validate_walk_settings
        validate_walk_settings(args)


def _random_features(ids, dimensions, seed, ternary=False):
    rng = np.random.default_rng(seed)
    order = np.argsort(np.asarray(ids), kind="stable")
    result = np.empty((len(ids), dimensions), dtype=np.float32)
    for start in range(0, len(ids), 32768):
        block = order[start:start + 32768]
        shape = (len(block), dimensions)
        if ternary:
            draws = rng.integers(6, size=shape, dtype=np.int8)
            result[block] = ((draws == 0).astype(np.float32) - (draws == 1)) * np.float32(math.sqrt(3 / dimensions))
        else:
            result[block] = rng.standard_normal(size=shape, dtype=np.float32) / np.float32(math.sqrt(dimensions))
    return result


def _transition(adjacency):
    degree = np.diff(adjacency.indptr)
    transition = adjacency.astype(np.float32)
    transition.data = np.repeat((1. / np.maximum(1, degree)).astype(np.float32), degree)
    return transition, degree


def diffusion(dataset, args):
    fast = args.training_method == "fastrp"
    adjacency = dataset.adjacency(False) if fast else dataset.adjacency(True).T.tocsr()
    transition, degree = _transition(adjacency)
    identity = _random_features(dataset.concept_ids, args.embedding_dim, args.seed, ternary=fast)
    if fast:
        scales = np.zeros(len(degree), dtype=np.float32)
        nonzero = degree > 0
        scales[nonzero] = (degree[nonzero] / degree.sum()) ** args.degree_exponent
        moving = identity * scales[:, None]
        result = np.zeros_like(moving)
        for _ in range(args.diffusion_steps):
            moving = transition @ moving
            result += moving
        return result / args.diffusion_steps, {"algorithm": "FastRP", "walk_direction": "undirected",
               "hop_weights": [1 / args.diffusion_steps] * args.diffusion_steps, "projection_density": 1/3,
               "degree_exponent": args.degree_exponent}
    moving = identity
    mass = np.ones(len(identity), dtype=np.float32)
    weighted = np.zeros_like(identity)
    total_mass = np.zeros_like(mass)
    for hop in range(args.diffusion_steps):
        moving = transition @ moving
        mass = transition @ mass
        weight = args.ancestor_decay ** hop
        weighted += weight * moving
        total_mass += weight * mass
    np.divide(weighted, total_mass[:, None], out=weighted, where=total_mass[:, None] > 0)
    weighted[total_mass == 0] = identity[total_mass == 0]
    return (1 - args.ancestor_mix) * identity + args.ancestor_mix * weighted, {
        "algorithm": "all-parent diffusion with a concept identity residual", "walk_direction": "child_to_parent",
        "ancestor_mix": args.ancestor_mix, "ancestor_decay": args.ancestor_decay, "diffusion_steps": args.diffusion_steps}


def _svd(operator, rank, seed, iterations):
    n = operator.shape[0]
    rank = min(rank, n)
    if n <= 256:
        left, singular, right = np.linalg.svd(operator.matmat(np.eye(n)), full_matrices=False)
    else:
        width = min(n, rank + 8)
        rng = np.random.default_rng(seed)
        probe = rng.normal(size=(n, width))
        def orthogonal(matrix):
            return np.linalg.qr(matrix, mode="reduced")[0]
        basis = orthogonal(operator.matmat(probe))
        for _ in range(iterations):
            basis = orthogonal(operator.matmat(orthogonal(operator.rmatmat(basis))))
        small, singular, right = np.linalg.svd(operator.rmatmat(basis).T, full_matrices=False)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            left = basis @ small
        if not np.isfinite(left).all():
            raise ValueError("Nonfinite singular-vector projection")
    usable = min(rank, int(np.count_nonzero(singular > singular[0] * n * np.finfo(float).eps)))
    left, right = left[:, :usable], right[:usable].T
    signs = np.sign(left[np.argmax(np.abs(left), axis=0), np.arange(usable)])
    scale = np.sqrt(singular[:usable]) * signs
    return left * scale, right * scale


def katz_operator(adjacency, beta):
    n = adjacency.shape[0]
    sources, targets = adjacency.nonzero()
    order = np.arange(n)
    if np.any(sources >= targets):
        graph = rx.PyDiGraph(multigraph=False)
        graph.add_nodes_from(range(n))
        for start in range(0, len(sources), 32768):
            block = slice(start, start + 32768)
            graph.add_edges_from_no_data(list(zip(sources[block], targets[block])))
        try:
            order = np.asarray(rx.topological_sort(graph), dtype=np.intp)
        except rx.DAGHasCycle as error:
            raise ValueError("This taxonomy Katz solver requires a directed acyclic parent-to-child graph") from error
    upper = adjacency[order][:, order].tocsr().astype(np.float64)
    lower = upper.T.tocsr()
    system = (sparse.eye(n, format="csr") - beta * upper).tocsr()
    transpose = system.T.tocsr()

    def apply(block, reverse):
        matrix, triangular = (lower, transpose) if reverse else (upper, system)
        rhs = beta * (matrix @ block[order])
        result = np.empty_like(rhs)
        result[order] = spsolve_triangular(triangular, rhs, lower=reverse, unit_diagonal=True)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite Katz proximity. Reduce katz_beta.")
        return result

    return LinearOperator((n, n), dtype=np.float64, matvec=lambda x: apply(x, False), rmatvec=lambda x: apply(x, True),
                          matmat=lambda x: apply(x, False), rmatmat=lambda x: apply(x, True))


def laplacian_coordinates(adjacency, args):
    n = adjacency.shape[0]
    degree = np.diff(adjacency.indptr)
    active = np.flatnonzero(degree)
    points = np.zeros((n, args.embedding_dim), dtype=np.float32)
    if not len(active):
        return points, {"algorithm": "Laplacian Eigenmaps", "effective_dimensions": 0, "eigenvalues": []}
    sub = adjacency[active][:, active]
    count, membership = csgraph.connected_components(sub, directed=False)
    rank = min(args.embedding_dim, len(active) - count)
    degrees = degree[active].astype(float)
    inv = sparse.diags(1 / np.sqrt(degrees))
    normalized = (inv @ sub @ inv).tocsr()
    volumes = np.bincount(membership, weights=degrees)
    constants = sparse.csr_matrix((np.sqrt(degrees / volumes[membership]), (np.arange(len(active)), membership)),
                                  shape=(len(active), count))
    def apply(block):
        return normalized @ block - 3 * (constants @ (constants.T @ block))
    if len(active) <= 256:
        eigenvalues, vectors = np.linalg.eigh(apply(np.eye(len(active))))
    else:
        operator = LinearOperator(sub.shape, matvec=apply, matmat=apply, dtype=np.float64)
        try:
            eigenvalues, vectors = eigsh(operator, k=rank, which="LA", tol=args.spectral_tolerance,
                                         v0=np.random.default_rng(args.seed).normal(size=len(active)),
                                         maxiter=args.spectral_iterations)
        except ArpackNoConvergence as error:
            raise ValueError(f"Laplacian eigensolver converged on {len(error.eigenvalues)}/{rank} axes. Increase spectral_iterations or relax spectral_tolerance.") from error
    take = np.argsort(eigenvalues)[::-1][:rank]
    values, vectors = eigenvalues[take], vectors[:, take]
    residual = np.linalg.norm(apply(vectors) - vectors * values, axis=0)
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(rank)])
    points[active, :rank] = vectors * signs / np.sqrt(degrees[:, None])
    return points, {"algorithm": "Laplacian Eigenmaps", "effective_dimensions": rank, "components_with_edges": int(count),
                    "eigenvalues": (1 - values).tolist(), "max_eigen_residual": float(residual.max())}


def fit_classical(dataset, args, root, stage):
    method, n, d = args.training_method, dataset.num_nodes, args.embedding_dim
    if n == 0:
        raise ValueError("A nonempty graph is required")
    if method == "random":
        points = _random_features(dataset.concept_ids, d, args.seed)
        details = {"algorithm": "Gaussian concept-ID control", "uses_edges_for_coordinates": False}
    elif method in ("fastrp", "ancestor"):
        points, details = diffusion(dataset, args)
    elif method == "laplacian":
        points, details = laplacian_coordinates(dataset.adjacency(False), args)
    elif method in ("deepwalk", "node2vec"):
        from context.walks import walk_coordinates
        points, details = walk_coordinates(dataset, args, stage)
    else:
        directed = method == "hope"
        adjacency = dataset.adjacency(directed)
        operator = katz_operator(adjacency, args.katz_beta) if directed else aslinearoperator(adjacency.astype(float))
        width = d // 2 if directed else d
        left, right = _svd(operator, width, args.seed, args.svd_iterations)
        rank = left.shape[1]
        points = np.zeros((n, d), dtype=np.float32)
        points[:, :rank] = left
        details = {"algorithm": "HOPE-Katz" if directed else "adjacency SVD", "singular_rank": rank,
                   "solver": "exact" if n <= 256 else "randomized block SVD", "power_iterations": args.svd_iterations}
        if directed:
            points[:, width:width + rank] = right
            details.update(katz_beta=args.katz_beta, role_width=width, directed_orientation="parent_to_child")
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"{method} produced nonfinite coordinates. Check the graph and method parameters.")
    anchor = root if args.euclidean_root == "fixed" else None
    origin = points[anchor].copy() if anchor is not None else np.zeros(d, dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        model = create_embedding(n, d, "euclidean", args.curvature, anchor, args.euclidean_distance_scale)
    with torch.no_grad():
        for start in range(0, n, 32768):
            block = slice(start, start + 32768)
            model.coordinates()[block].copy_(torch.from_numpy((points[block] - origin) / model.distance_scale))
    details.update(native_origin=origin.tolist(), numpy_version=np.__version__, scipy_version=scipy.__version__,
                   compute_device="cpu")
    return model.eval(), details
