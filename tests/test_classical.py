import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import rustworkx as rx
from scipy import sparse
import torch

from context.args import Args
from context.classical import METHODS, _random_features, diffusion, fit_classical, katz_operator, laplacian_coordinates
from context.dataset import GraphEdges
from context.export import load_embedding
from context.graph import Graph
from context.selection import split_edges
from context.train import output_folder, train
from context.walks import WalkCorpus


def sample():
    graph = rx.PyDiGraph()
    graph.add_nodes_from([101, 107, 113, 127, 131, 137])
    graph.add_edges_from_no_data([(0, 2), (0, 1), (1, 3), (2, 3), (3, 4), (1, 4)])
    return GraphEdges(graph, directed=True)


def setup(method, **settings):
    return replace(Args(geometry="euclidean", training_method=method, embedding_dim=4, seed=17, threads=1,
                        walks_per_node=2, walk_length=8, walk_window=2, walk_epochs=2), **settings)


def test_katz_and_asymmetric_role_scores(tmp_path):
    dataset = sample()
    a = dataset.adjacency(True).toarray().astype(float)
    order = [5, 2, 4, 1, 3, 0]
    permuted = a[np.ix_(order, order)]
    beta = .3
    expected = np.linalg.inv(np.eye(len(a)) - beta * permuted) - np.eye(len(a))
    operator = katz_operator(sparse.csr_matrix(permuted), beta)
    np.testing.assert_allclose(operator @ np.eye(len(a)), expected, atol=1e-12)
    np.testing.assert_allclose(operator.T @ np.eye(len(a)), expected.T, atol=1e-12)
    model, details = fit_classical(dataset, setup("hope", embedding_dim=13, katz_beta=beta,
                                                   euclidean_distance_scale=2.), 0, tmp_path)
    native = model.coordinates().detach().numpy() * 2 + np.asarray(details["native_origin"])
    sources, targets = native[:, :6], native[:, 6:12]
    np.testing.assert_allclose(sources @ targets.T, np.linalg.inv(np.eye(len(a)) - beta * a) - np.eye(len(a)), atol=2e-7)
    assert np.all(native[:, -1] == 0)
    with pytest.raises(ValueError, match="acyclic"):
        katz_operator(sparse.csr_matrix([[0, 1], [1, 0]]), beta)


def test_adjacency_svd_reproduces_dense_row_geometry(tmp_path):
    data = sample()
    args = setup("adjacency_svd", embedding_dim=8, euclidean_root="trainable")
    model, _ = fit_classical(data, args, 0, tmp_path)
    points = model.coordinates().detach().numpy()
    a = data.adjacency(False).toarray().astype(float)
    u, sigma, _ = np.linalg.svd(a)
    np.testing.assert_allclose(points @ points.T, (u * sigma) @ u.T, atol=3e-7)


def test_laplacian_generalized_eigen_equation_with_isolate():
    a = sample().adjacency(False)
    points, receipt = laplacian_coordinates(a, setup("laplacian", embedding_dim=6))
    d = np.diag(np.diff(a.indptr))
    active = points[:, :receipt["effective_dimensions"]]
    np.testing.assert_allclose((d - a.toarray()) @ active, d @ active * receipt["eigenvalues"], atol=2e-7)
    np.testing.assert_allclose(active.T @ d @ active, np.eye(active.shape[1]), atol=2e-7)
    assert receipt["max_eigen_residual"] < 1e-8 and not points[5].any()
    blocks = sparse.block_diag([a, a], format="csr")
    _, record = laplacian_coordinates(blocks, setup("laplacian", embedding_dim=15))
    assert record["components_with_edges"] == 2 and record["effective_dimensions"] == 8


def test_fastrp_matches_transition_powers_and_ancestor_uses_both_parents():
    data = sample()
    opts = setup("fastrp", diffusion_steps=3, degree_exponent=-.4)
    adjacency = data.adjacency(False).toarray().astype(float)
    degrees = adjacency.sum(1)
    p = adjacency / np.maximum(1, degrees)[:, None]
    scale = np.zeros_like(degrees)
    scale[degrees > 0] = (degrees[degrees > 0] / degrees.sum()) ** opts.degree_exponent
    codes = _random_features(data.concept_ids, opts.embedding_dim, opts.seed, ternary=True)
    answer = sum(np.linalg.matrix_power(p, k) for k in [1, 2, 3]) / 3 @ (codes * scale[:, None])
    np.testing.assert_allclose(diffusion(data, opts)[0], answer, atol=3e-7)
    gauss = _random_features(data.concept_ids, 4, 17)
    pure = setup("ancestor", ancestor_mix=1., diffusion_steps=1)
    pooled = diffusion(data, pure)[0]
    np.testing.assert_allclose(pooled[3], (gauss[1] + gauss[2]) / 2, atol=1e-7)
    np.testing.assert_array_equal(pooled[[0, 5]], gauss[[0, 5]])
    opts = replace(pure, ancestor_mix=0.)
    np.testing.assert_array_equal(diffusion(data, opts)[0], gauss)
    np.testing.assert_array_equal(gauss[::-1], _random_features(data.concept_ids[::-1], 4, 17))


def test_walk_transition_bias_and_fixed_corpus():
    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from_no_data([(0, 1), (0, 2), (1, 2), (1, 3)])
    a = GraphEdges(graph).adjacency(False)
    opts = setup("node2vec", return_p=2., inout_q=.5)
    corpus = WalkCorpus(a, opts)
    rows = np.ones(60000, dtype=np.int64)
    neighbors = corpus.step(rows * 0, rows, np.random.default_rng(17))
    frequencies = np.bincount(neighbors, minlength=4)[[0, 2, 3]] / len(rows)
    np.testing.assert_allclose(frequencies, np.array([.5, 1., 2.]) / 3.5, atol=.012)
    walks = list(corpus)
    assert len(walks) == 8 and walks == list(corpus)
    assert all(a[u, v] for walk in walks for u, v in zip(walk, walk[1:]))
    uniform = list(WalkCorpus(a, replace(opts, return_p=1., inout_q=1.)))
    assert uniform == list(WalkCorpus(a, replace(opts, training_method="deepwalk")))


@pytest.fixture
def graph_file(tmp_path):
    children = np.arange(1, 32)
    links = sorted(set(zip((children - 1) // 2, children)) | set(zip(children // 4, children)))
    graph = Graph(pl.DataFrame([(int(s), int(t), 1) for s, t in links], schema=["parent", "child", "weight"], orient="row"))
    path = tmp_path / "graph.pkl"
    graph.save(path)
    return path


@pytest.mark.parametrize("method", METHODS)
def test_normal_workflow_fit_only_training_edges_and_refit(method, graph_file, tmp_path):
    if method in ("deepwalk", "node2vec"):
        pytest.importorskip("gensim")
    opts = setup(method, graph_file=str(graph_file), graph_id="full", experiment_id="selection", root_node_label=0,
                 output_directory=str(tmp_path), directed=True, validation_fraction=.08, retrieval_direction="parents",
                 eval_interval=1, split_policy="preserve_degrees")
    train(opts)
    folder = output_folder(opts, "euclidean")
    result = load_embedding(folder / "embedding.pt")
    record = json.loads((folder / "training.json").read_text())
    assert len(result["concept_ids"]) == 32 and result["metadata"]["training_method"] == method
    assert result["metadata"]["graph_root_index"] == result["root_index"] and record["test"]["queries"] > 0
    raw = GraphEdges(Graph.load(graph_file).full_graph, directed=True)
    with np.load(folder / "partitions.npz") as saved:
        expected_split = split_edges(raw, opts)
        np.testing.assert_array_equal(saved["train"], expected_split.training)
        fitting = raw.with_training_edges(saved["train"])
    check = tmp_path / "oracle"
    check.mkdir()
    independent, _ = fit_classical(fitting, opts, raw.concept_ids.index(0), check)
    torch.testing.assert_close(result["embeddings"], independent.coordinates(), rtol=0, atol=0)
    train(replace(opts, resume=True))
    train(Args(output_directory=str(tmp_path), experiment_id="full-refit", graph_file=str(graph_file), refit_from=str(folder)))
    refit = json.loads((output_folder(replace(opts, experiment_id="full-refit"), "euclidean") / "training.json").read_text())
    assert refit["graph_edges"] == refit["training_edges"] == raw.num_edges and refit["training_scope"] == "full_graph_refit"


@pytest.mark.parametrize("method", ["deepwalk", "node2vec"])
def test_walk_epoch_recovery(method, graph_file, tmp_path, monkeypatch):
    Word2Vec = pytest.importorskip("gensim.models").Word2Vec
    opts = setup(method, graph_file=str(graph_file), graph_id="full", root_node_label=0, output_directory=str(tmp_path),
                 experiment_id="uninterrupted", walk_epochs=3, checkpoint_interval=1, eval_interval=0)
    train(opts)
    original = Word2Vec.train
    def interrupt(self, *arguments, **settings):
        if self.context_walk_epoch == 1:
            raise InterruptedError("after epoch one")
        return original(self, *arguments, **settings)
    monkeypatch.setattr(Word2Vec, "train", interrupt)
    other = replace(opts, experiment_id="resumed")
    with pytest.raises(InterruptedError):
        train(other)
    assert list(output_folder(other, "euclidean").glob(".context-fit-*/walk.model"))
    monkeypatch.setattr(Word2Vec, "train", original)
    train(replace(other, resume=True))
    def values(settings):
        return load_embedding(output_folder(settings, "euclidean") / "embedding.pt")["embeddings"]
    torch.testing.assert_close(values(opts), values(other), rtol=0, atol=0)


def test_sparse_spectral_solvers_and_laplacian_diagnostics(tmp_path):
    n = 360
    graph = rx.PyDiGraph()
    graph.add_nodes_from(list(range(n)))
    graph.add_edges_from_no_data([(i // 2, i) for i in range(1, n)])
    graph.add_edges_from_no_data([(i // 4, i) for i in range(10, n, 4)])
    data = GraphEdges(graph)
    for method in ("adjacency_svd", "hope", "laplacian"):
        model, details = fit_classical(data, setup(method), 0, tmp_path)
        assert torch.isfinite(model.coordinates()).all() and model.coordinates().abs().sum() > 0
        if method == "laplacian":
            assert details["max_eigen_residual"] < 5e-5
        else:
            assert details["solver"] == "randomized block SVD"


def test_permutation_control_preserves_points_and_training_degree_strata(tmp_path, graph_file):
    from context.controls import permute_embedding
    opts = setup("random", graph_file=str(graph_file), graph_id="full", root_node_label=0, output_directory=str(tmp_path),
                 experiment_id="original", validation_fraction=.10, directed=True, eval_interval=1)
    train(opts)
    folder = output_folder(opts, "euclidean")
    initial = load_embedding(folder / "embedding.pt")
    with np.load(folder / "partitions.npz") as parts:
        src, dst = parts["train"].T
    rows = len(initial["concept_ids"])
    parents = np.bincount(dst, minlength=rows)
    child_bin = np.floor(np.log2(1 + np.bincount(src, minlength=rows)))
    for mode in ("none", "degree"):
        destination = tmp_path / ("control-" + mode)
        record = permute_embedding(folder / "embedding.pt", destination, matching=mode, graph_file=graph_file, seed=9, write_tsv=True)
        shuffled = load_embedding(destination / "embedding.pt")
        indices = np.load(destination / "permutation.npy")
        np.testing.assert_array_equal(np.sort(indices), np.arange(rows))
        assert indices[initial["root_index"]] == initial["root_index"]
        torch.testing.assert_close(initial["concept_ids"], shuffled["concept_ids"])
        torch.testing.assert_close(initial["embeddings"][indices], shuffled["embeddings"], rtol=0, atol=0)
        assert record["rows_with_changed_vector"] == record["rows_reassigned"] > 0
        if mode == "degree":
            np.testing.assert_array_equal(parents[indices], parents)
            np.testing.assert_array_equal(child_bin[indices], child_bin)
        with pytest.raises(FileExistsError):
            permute_embedding(folder / "embedding.pt", destination)


def test_randomized_svd_matches_a_low_rank_operator():
    from scipy.sparse.linalg import aslinearoperator
    from context.classical import _svd
    rng = np.random.default_rng(191)
    a, b = (rng.normal(size=(310, 4)) for _ in range(2))
    target = np.einsum("ik,jk->ij", a, b, optimize=False)
    left, right = _svd(aslinearoperator(target), 4, 6, 2)
    np.testing.assert_allclose(np.einsum("ik,jk->ij", left, right, optimize=False), target, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_single_root_without_training_edges(method, tmp_path):
    if method in ("deepwalk", "node2vec"):
        pytest.importorskip("gensim")
    graph = rx.PyDiGraph()
    graph.add_node(39)
    model, _ = fit_classical(GraphEdges(graph), setup(method), 0, tmp_path)
    assert model.coordinates().shape == (1, 4) and torch.count_nonzero(model.coordinates()) == 0


def test_euclidean_method_rejects_incompatible_geometry():
    from context.train import _validate
    with pytest.raises(ValueError, match="--geometry euclidean"):
        _validate(setup("node2vec", geometry="hyperbolic"))
    with pytest.raises(ValueError, match="ancestor_mix"):
        _validate(setup("ancestor", ancestor_mix=1.1))
