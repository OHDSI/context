import json
from dataclasses import replace
from importlib import import_module

import numpy as np
import polars as pl
import pytest
import rustworkx as rx
import torch
from scipy.spatial.distance import cdist

from context.args import Args
from context.dataset import GraphEdges, GraphEmbeddingDataset
from context.export import load_embedding
from context.graph import Graph
from context.landmarks import LandmarkDistances, fit_coordinates, landmark_distances
from context.model import pairwise_distance
from context.train import output_folder, train


def edges(n, pairs):
    graph = rx.PyDiGraph()
    graph.add_nodes_from(list(range(n)))
    graph.add_edges_from_no_data(pairs)
    return GraphEdges(graph, directed=True)


@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic"])
@pytest.mark.parametrize("anchor", [1, None])
def test_exact_metric_recovery_with_out_of_sample_points(geometry, anchor):
    points = np.array([[.1, .2], [-.1, 0], [.1, -.2], [.2, .1], [-.15, .22], [.03, .17]])
    distances = cdist(points, points)
    curvature = 2.5
    if geometry == "hyperbolic":
        margins = 1 - curvature * np.square(points).sum(axis=1)
        distances = 2 / np.sqrt(curvature) * np.arcsinh(np.sqrt(curvature) * distances / np.sqrt(margins[:, None] * margins))
    rows = np.array([0, 1, 2, 4])
    basis = LandmarkDistances(rows, distances[rows], 1)
    args = Args(geometry=geometry, embedding_dim=4, curvature=curvature, landmark_edge_length=1.,
                euclidean_distance_scale=2., euclidean_root="fixed" if anchor is not None else "trainable")
    model, detail = fit_coordinates(basis, args, anchor)
    result = model.coordinates().detach().double()
    out = pairwise_distance(result, result, model.curvature)
    if geometry == "euclidean":
        out *= model.distance_scale
    np.testing.assert_allclose(out.numpy(), distances, atol=2e-6)
    assert detail["effective_dimensions"] == 2 and np.count_nonzero(result[:, 2:].numpy()) == 0
    if anchor is not None:
        assert torch.count_nonzero(result[anchor]) == 0


def test_all_parents_components_and_isolates():
    data = edges(7, [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5)])
    result = landmark_distances(data, 0, 20, 3)
    actual = result.hops[np.argsort(result.rows)]
    assert result.components == 3 and actual[3, 2] == actual[3, 1] == 1
    assert actual[1, 2] == actual[0, 4] == actual[0, 6] == 2 and actual[5, 6] == 3
    np.testing.assert_array_equal(result.hops, landmark_distances(data, 0, 20, 3).hops)
    np.testing.assert_array_equal(actual, actual.T)


@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic"])
def test_singleton_and_depth_scaling(geometry):
    singleton = landmark_distances(edges(1, []), 0, 3, 23)
    model, info = fit_coordinates(singleton, Args(geometry=geometry, embedding_dim=5), 0)
    assert info["effective_dimensions"] == 0 and info["distinct_coordinate_rows"] == 1 and torch.count_nonzero(model.coordinates()) == 0
    path = landmark_distances(edges(300, [(i, i + 1) for i in range(299)]), 60, 9, 23)
    opts = Args(geometry=geometry, embedding_dim=3, curvature=.5)
    model, info = fit_coordinates(path, opts, 60)
    points = model.coordinates().detach().double()
    indices = torch.tensor([0, 60, 104, 299])
    recovered = pairwise_distance(points[indices], points[indices], model.curvature).numpy()
    expected = np.abs(indices[:, None].numpy() - indices.numpy()) * info["edge_length"]
    np.testing.assert_allclose(recovered, expected, atol=4e-5)
    if geometry == "hyperbolic":
        with pytest.raises(ValueError, match="Reduce landmark_edge_length"):
            fit_coordinates(path, replace(opts, landmark_edge_length=500.), 60)


def test_held_out_fit_matches_training_only_oracle_and_full_refit(tmp_path, monkeypatch):
    children = np.arange(1, 40)
    pairs = sorted(set(zip((children - 1) // 2, children)) | set(zip(children // 4, children)))
    graph = Graph(pl.DataFrame([(int(s), int(t), 1) for s, t in pairs], schema=["p", "c", "w"], orient="row"))
    folder = tmp_path / "selection"
    folder.mkdir()
    graph.save(folder / "graph.pkl")
    args = Args(output_directory=str(tmp_path), experiment_id="selection", graph_id="full", root_node_label=0,
                geometry="both", training_method="landmark", validation_fraction=.12, split_policy="preserve_degrees",
                retrieval_direction="parents", eval_interval=1, eval_queries=0, seed=41, landmarks=9,
                negative_samples=0, euclidean_root="trainable", threads=1)
    def reject(*unused, **options):
        pytest.fail("negative sampler constructed")
    monkeypatch.setattr(GraphEmbeddingDataset, "__init__", reject)
    train(args)
    for geometry in ("hyperbolic", "euclidean"):
        directory = output_folder(args, geometry)
        saved = json.loads((directory / "training.json").read_text())
        values = load_embedding(directory / "embedding.pt")
        with np.load(directory / "partitions.npz") as parts:
            training = GraphEdges(graph.full_graph).with_training_edges(parts["train"])
            assert len(parts["test"]) and len(parts["validation"])
        root = training.concept_ids.index(0)
        basis = landmark_distances(training, root, 9, 41)
        oracle, _ = fit_coordinates(basis, replace(args, geometry=geometry), root)
        torch.testing.assert_close(values["embeddings"], oracle.coordinates().detach(), rtol=0, atol=0)
        assert saved["test"]["queries"] and values["metadata"]["selected_epoch"] is None
        assert values["root_index"] == (root if geometry == "hyperbolic" else None)
        if geometry == "euclidean":
            assert torch.count_nonzero(values["embeddings"][root]) > 0
    euclidean = output_folder(args, "euclidean")
    train(Args(output_directory=str(tmp_path), experiment_id="all", refit_from=str(euclidean)))
    refitted = json.loads((output_folder(replace(args, experiment_id="all"), "euclidean") / "training.json").read_text())
    assert refitted["training_edges"] == len(pairs) and refitted["training_scope"] == "full_graph_refit"
    assert refitted["selection_identity"]["graph_sha256"] == saved["graph_sha256"]
    monkeypatch.setattr(import_module("context.landmarks"), "landmark_distances", reject)
    train(replace(args, resume=True))
    with (euclidean / "embedding.pt").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="artifact has changed"):
        train(replace(args, resume=True))


def test_small_complete_graph_and_independent_output(tmp_path):
    graph = Graph(pl.DataFrame([(11, 13, 1)], schema=["p", "c", "w"], orient="row"))
    path = tmp_path / "graph.pkl"
    graph.save(path)
    args = Args(graph_file=str(path), graph_id="full", root_node_label=11, output_directory=str(tmp_path),
                experiment_id="twonodes", training_method="landmark", geometry="both", eval_interval=1)
    results = train(args)
    assert [r["map_score"] for r in results] == [1., 1.]
    assert output_folder(args, "hyperbolic") != output_folder(replace(args, training_method="sampled"), "hyperbolic")
    with pytest.raises(ValueError, match="--device cpu"):
        train(replace(args, device="mps"))

def test_unresolved_siblings_are_counted():
    graph = edges(9, [(0, i) for i in range(1, 9)])
    basis = landmark_distances(graph, 0, 2, 3)
    model, info = fit_coordinates(basis, Args(geometry="euclidean", embedding_dim=4), 0)
    _, counts = np.unique(model.coordinates().detach().numpy(), axis=0, return_counts=True)
    assert info["distinct_coordinate_rows"] == len(counts)
    assert info["nodes_sharing_coordinates"] == sum(counts[counts > 1]) >= 7
    assert info["largest_tied_group"] == max(counts)
    assert info["distinct_coordinate_rows"] <= info["distinct_distance_profiles"]
    for rows in np.split(np.argsort(basis.distance_profiles[1], kind="stable"), np.cumsum(basis.distance_profiles[2])[:-1]):
        values = model.coordinates().detach().numpy()[rows]
        assert np.all(values == values[0])


def test_reciprocal_edges_stay_in_one_fold():
    from context.selection import split_edges
    data = edges(7, [(a, b) for a in range(7) for b in range(7) if a != b])
    for directed in (True, False):
        split = split_edges(data, Args(directed=directed, validation_fraction=.15, split_seed=25))
        groups = [{tuple(sorted(edge)) for edge in fold} for fold in (split.training, split.validation, split.test)]
        assert len(set.union(*groups)) == 21 and sum(map(len, groups)) == 21


def test_resume_interrupted_landmark_publication(tmp_path, monkeypatch):
    graph = Graph(pl.DataFrame([(11, 12, 1), (12, 13, 1)], schema=["p", "c", "w"], orient="row"))
    path = tmp_path / "graph.pkl"
    graph.save(path)
    args = Args(graph_file=str(path), graph_id="full", root_node_label=11, output_directory=str(tmp_path),
                training_method="landmark", eval_interval=0)
    module = import_module("context.train")
    publish_file = module.publish_file
    def interrupt(source, destination):
        if source.name == "training.json":
            raise InterruptedError("publication")
        publish_file(source, destination)
    monkeypatch.setattr(module, "publish_file", interrupt)
    with pytest.raises(InterruptedError):
        train(args)
    monkeypatch.setattr(module, "publish_file", publish_file)
    def forbid(*args):
        pytest.fail("completed geometry recalculated")
    monkeypatch.setattr(import_module("context.landmarks"), "landmark_distances", forbid)
    train(replace(args, resume=True))
    assert len(load_embedding(output_folder(args, "hyperbolic") / "embedding.pt")["concept_ids"]) == 3
