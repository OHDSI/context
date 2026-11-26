import json
import pickle
import sqlite3
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rustworkx as rx
import torch
from scipy.spatial.distance import cdist

from context.args import Args
from context.dataset import GraphEmbeddingDataset
from context.export import file_sha, load_embedding
from context.graph import Graph
from context.loss import candidate_loss
from context.model import create_embedding
from context.optim import make_optimizer
from context.recovery import completed_records
from context.retrieval import edge_index, query_rows, rank_neighbors
from context.selection import split_edges
from context.train import build_graph, output_folder, train


@pytest.fixture
def options(tmp_path):
    children = np.arange(1, 21)
    edges = sorted(set(zip((children - 1) // 2, children)) | set(zip(children // 4, children)))
    graph = Graph(pl.DataFrame([(int(s + 101), int(t + 101), "subsumes") for s, t in edges],
                               schema=["parent", "child", "type"], orient="row"))
    graph.ancestral_subgraph([int(v + 101) for v in children])
    path = tmp_path / "graph.pkl"
    graph.save(path)
    return Args(output_directory=str(tmp_path), experiment_id="fixture", graph_file=str(path), seed=12,
                root_node_label=101, geometry="euclidean", burn_in=False, learning_rate=.01,
                optimizer="adam", epochs=4, eval_interval=1, batch_size=9, negative_samples=3,
                embedding_dim=4, save_top=1, checkpoint_interval=2, eval_queries=3, directed=True)


def test_all_complement_ranks_including_hubs_and_isolates():
    graph = rx.PyDiGraph()
    graph.add_nodes_from(list(range(82)))
    graph.remove_node(6)
    graph.add_edges_from_no_data([(0, t) for t in range(1, 82) if t not in (6, 10, 32, 81)] +
                                 [(1, 2), (1, 4), (1, 80), (3, 1), (0, 3), (0, 0)])
    dataset = GraphEmbeddingDataset(graph, directed=True)
    for source in (0, 1, 3):
        adjacency = dataset.adjacency(True)
        neighbors = set(adjacency[source].indices)
        expected = sorted(set(range(dataset.num_nodes)) - neighbors - {source})
        class AllRanks:
            def integers(self, high, size):
                assert high.item() == len(expected)
                return np.arange(len(expected))[None, :]
        dataset.num_negative_samples = len(expected)
        actual = dataset._sample(np.array([[source, next(iter(neighbors))]]), AllRanks())
        assert actual.tolist() == [expected]


@pytest.mark.parametrize("geometry", ["hyperbolic", "euclidean"])
def test_tiled_ranking_matches_exhaustive_oracle_with_known_positives(geometry):
    points = np.random.default_rng(41).normal(0, .04, (37, 5))
    points[5] = points[4]
    points[10] = points[3]
    targets = [(i, j) for i in range(6) for j in (7, 10, 12)] + [(0, 4), (0, 27)]
    known = [(i, j) for i in range(6) for j in (4, 6, 18)]
    positive_index, known_index = (edge_index(pairs, len(points)) for pairs in (targets, known))
    queries = query_rows(positive_index, 0, 3)
    curvature = 2.5 if geometry == "hyperbolic" else None
    distances = cdist(points, points)
    if curvature:
        margin = 1 - curvature * np.sum(points ** 2, axis=1)
        distances = 2 / np.sqrt(curvature) * np.arcsinh(distances * np.sqrt(curvature / np.outer(margin, margin)))
    all_filtered, all_ap, all_raw = [], [], []
    for source in queries:
        positive = positive_index[source].indices
        excluded = (set(known_index[source].indices) | {source}) - set(positive)
        candidates = np.array(sorted(set(range(len(points))) - excluded))
        negatives = np.setdiff1d(candidates, positive)
        row = distances[source]
        ap, raw = [], []
        for target in positive:
            threshold = row[target]
            all_filtered.append(1 + sum(row[negatives] < threshold) + .5 * sum(row[negatives] == threshold))
            raw.append(sum(row[candidates] < threshold) + .5 * (1 + sum(row[candidates] == threshold)))
            ap.append(sum(row[positive] <= threshold) / sum(row[candidates] <= threshold))
        all_ap.append(np.mean(ap))
        all_raw.append(np.mean(raw))
    result = rank_neighbors(points, positive_index, queries, curvature, known_index, batch_size=2, candidate_block=3)
    assert result["mean_rank"] == pytest.approx(np.mean(all_filtered))
    assert result["macro_average_precision"] == pytest.approx(np.mean(all_ap))
    assert result["macro_mean_positive_midrank"] == pytest.approx(np.mean(all_raw))
    assert len(query_rows(positive_index, 2, 17)) == 2


@pytest.mark.parametrize("geometry", ["hyperbolic", "euclidean"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_sgd_matches_dense_for_repeats_and_untouched_rows(geometry, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(13)
    dense, sparse = (create_embedding(103, 8, geometry, 2.5, 9).to(device) for _ in range(2))
    sparse.load_state_dict(dense.state_dict())
    original = sparse.coordinates().detach().clone()
    args = Args(geometry=geometry, optimizer="sgd")
    opts = [make_optimizer(model, replace(args, sgd_updates=mode), .06)
            for model, mode in ((dense, "dense"), (sparse, "sparse"))]
    for _ in range(5):
        for model, optimizer in zip([dense, sparse], opts):
            optimizer.zero_grad(set_to_none=True)
            candidate_loss(model(torch.tensor([[1], [3], [9], [1]], device=device),
                                 torch.tensor([[0, 2, 9], [4, 2, 9], [1, 3, 1], [4, 9, 7]], device=device))).backward()
            optimizer.step()
    torch.testing.assert_close(sparse.coordinates(), dense.coordinates(), atol=2e-7, rtol=1e-5)
    assert torch.equal(original[10:], sparse.coordinates()[10:])
    assert not torch.count_nonzero(sparse.coordinates()[9])


@pytest.mark.parametrize("geometry,device,optimizer,updates", [
    ("euclidean", "cpu", "adam", "auto"), ("hyperbolic", "cpu", "adam", "auto"),
    ("hyperbolic", "cpu", "sgd", "sparse"), ("hyperbolic", "mps", "adam", "auto")])
def test_resume_after_unsaved_epoch_replays_optimizer_rng_and_selection(options, monkeypatch, geometry, device, optimizer, updates):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    settings = replace(options, geometry=geometry, device=device, optimizer=optimizer, sgd_updates=updates,
                       burn_in=True, burn_in_epochs=2, validation_fraction=.12, retrieval_direction="parents")
    reference = replace(settings, experiment_id="reference")
    train(reference)
    module = import_module("context.train")
    save = module.save_model
    def interrupt(model, args, folder, epoch, *remaining, **kwargs):
        if epoch == 4:
            raise KeyboardInterrupt
        save(model, args, folder, epoch, *remaining, **kwargs)
    monkeypatch.setattr(module, "save_model", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(settings)
    with pytest.raises(ValueError, match="matches"):
        train(replace(settings, seed=123, resume=True))
    monkeypatch.setattr(module, "save_model", save)
    train(replace(settings, resume=True))
    first, replay = (output_folder(config, geometry) for config in (reference, settings))
    first_summary, replay_summary = [json.loads((p / "training.json").read_text()) for p in (first, replay)]
    assert first_summary["selected_epoch"] == replay_summary["selected_epoch"]
    np.testing.assert_allclose(first_summary["training_loss"], replay_summary["training_loss"],
                               rtol=1e-6 if device == "mps" else 0, atol=0)
    assert first_summary["learning_rates"] == replay_summary["learning_rates"]
    assert first_summary["test"] == replay_summary["test"]
    torch.testing.assert_close(load_embedding(first / "embedding.pt")["embeddings"],
                               load_embedding(replay / "embedding.pt")["embeddings"],
                               rtol=1e-5 if device == "mps" else 0, atol=2e-7 if device == "mps" else 0)
    assert not list(replay.glob(".context-fit-*"))
    monkeypatch.setattr(module, "_fit", lambda *args: pytest.fail("Completed fit reran"))
    train(replace(settings, resume=True))
    (replay / "labels.tsv").write_text("broken")
    with pytest.raises(ValueError, match="changed"):
        completed_records(replay, replay_summary["fit_identity"])


def test_validation_refit_applies_selected_trace_and_checks_graph(options):
    selection = replace(options, optimizer="sgd", validation_fraction=.12, retrieval_direction="parents",
                        euclidean_root="trainable")
    train(selection)
    folder = output_folder(selection, selection.geometry)
    metadata = json.loads((folder / "training.json").read_text())
    folds = np.load(folder / "partitions.npz")
    sets = [set(map(tuple, folds[name])) for name in ("train", "validation", "test")]
    assert sum(map(len, sets)) == len(set.union(*sets)) == metadata["graph_edges"]
    assert metadata["best_validation_ap"] == max(record["map_score"] for record in metadata["evaluation_records"])
    assert metadata["test"]["queries"] <= options.eval_queries
    args = replace(selection, refit_from=str(folder), experiment_id="refit")
    assert train(args) == []
    refit = output_folder(args, args.geometry)
    full = json.loads((refit / "training.json").read_text())
    assert full["training_edges"] == metadata["graph_edges"]
    assert full["training_scope"] == "full_graph_refit" and "test" not in full
    assert full["learning_rates"] == metadata["learning_rates"][:metadata["selected_epoch"]]
    assert full["selected_epoch"] == len(full["learning_rates"])
    assert load_embedding(refit / "embedding.pt")["metadata"]["selection_identity"]["selection_sha256"] == file_sha(folder / "training.json")
    changed = Path(options.output_directory) / "changed.pkl"
    Graph(pl.DataFrame({"p": [101, 102], "c": [102, 103], "type": ["subsumes"] * 2})).save(changed)
    with pytest.raises(ValueError, match="graph used for selection"):
        train(replace(args, graph_file=str(changed)))


def test_preserve_degrees_keeps_direction_and_groups_reciprocals(options):
    graph = rx.PyDiGraph()
    graph.add_nodes_from(list(range(25)))
    edges = [(i, j) for i in range(12) for j in range(12, 25)] + [(15, 0), (13, 6)]
    graph.add_edges_from_no_data(edges)
    dataset = GraphEmbeddingDataset(graph, directed=False)
    partition = split_edges(dataset, replace(options, directed=False, validation_fraction=.1, split_policy="preserve_degrees"))
    sets = [set(tuple(sorted(pair)) for pair in fold) for fold in (partition.training, partition.validation, partition.test)]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    for side in (0, 1):
        assert set(partition.training[:, side]) == set(np.array(edges)[:, side])
    tree = rx.PyDiGraph()
    tree.add_nodes_from(list(range(10)))
    tree.add_edges_from_no_data([(0, child) for child in range(1, 10)])
    with pytest.raises(ValueError, match="redundant"):
        split_edges(GraphEmbeddingDataset(tree), replace(options, split_policy="preserve_degrees", validation_fraction=.1))


def test_versioned_cache_reads_old_and_roundtrips_removed_nodes_loops_and_payloads(tmp_path):
    old = {"full_graph": {"nodes": ["z", "a", "b", "c"], "edges": [("a", "b", "first"), ("a", "b", {"a": 2}),
                                                                    ("c", "c", None), ("b", "a", 2)]}, "subgraphs": {}}
    original = tmp_path / "old.pkl"
    original.write_bytes(pickle.dumps(old))
    graph = Graph.load(original)
    graph.full_graph.remove_node(0)
    graph.full_graph.remove_edge_from_index(0)
    graph.full_graph.add_edge(1, 3, "new")
    graph.subgraph["copy"] = graph.full_graph.copy()
    graph.subgraph["copy"].add_node("isolated")
    path = tmp_path / "current.pkl"
    graph.save(path)
    restored = Graph.load(path)
    for before, after in ((graph.full_graph, restored.full_graph), (graph.subgraph["copy"], restored.subgraph["copy"])):
        assert list(before.nodes()) == list(after.nodes())
        assert [(before[s], before[t], value) for s, t, value in before.weighted_edge_list()] == [
            (after[s], after[t], value) for s, t, value in after.weighted_edge_list()]


def test_old_and_checksummed_embedding_readers(options):
    train(options)
    folder = output_folder(options, options.geometry)
    path = folder / "embedding.pt"
    data = load_embedding(path)
    old = folder / "old.pt"
    torch.save({k: v for k, v in data.items() if k != "metadata"}, old)
    assert torch.equal(load_embedding(old)["embeddings"], data["embeddings"])
    sidecar = path.with_suffix(".json")
    sidecar.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        load_embedding(path)
    old_payload = {"metadata": data["metadata"], "embeddings": data["embeddings"], "concept_ids": data["concept_ids"]}
    torch.save(old_payload, path)
    sidecar.write_text(json.dumps({**data["metadata"], "payload_sha256": file_sha(path)}))
    assert torch.equal(load_embedding(path)["concept_ids"], data["concept_ids"])
    with path.open("ab") as stream:
        stream.write(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_embedding(path)


def test_whole_rooted_taxonomy_and_changed_seed_cache(options, tmp_path):
    concepts, relationships, seeds = (tmp_path / name for name in ("concept.tsv", "rel.tsv", "ids.csv"))
    concepts.write_text("concept_id\tconcept_name\tvocabulary_id\tstandard_concept\tinvalid_reason\n" +
                        "".join(f"{i}\t{i}\tSNOMED\tS\t\n" for i in range(101, 109)))
    relationships.write_text("concept_id_1\tconcept_id_2\trelationship_id\tinvalid_reason\n" +
                             "".join(f"{c}\t{p}\tIs a\t\n" for p, c in [(101,102),(101,103),(102,104),(103,104),(102,105),(106,107)]))
    seeds.write_text("conceptId\n104\n")
    args = replace(options, graph_file=None, concept_map=str(concepts), relationship_file_path=str(relationships),
                   concept_id_file=str(seeds), epochs=1)
    entire = build_graph(replace(args, taxonomy_scope="subtree", concept_id_file="unused"))
    assert set(entire.full_graph.nodes()) == {101, 102, 103, 104, 105}
    for anchor in (None, 101):
        complete = build_graph(replace(args, taxonomy_scope="all", concept_id_file="unused", root_node_label=anchor))
        assert set(complete.full_graph.nodes()) == set(range(101, 109))
        assert complete.extraction["isolated_nodes"] == 1 and complete.extraction["scope"] == "all"
    train(args)
    seeds.write_text("conceptId\n105\n")
    with pytest.raises(ValueError, match="Graph inputs changed"):
        train(args)
    database = tmp_path / "direct.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("create table covariateRef(conceptId integer)")
        conn.execute("insert into covariateRef values(104)")
    assert Graph.get_concept_set(database, root_concept_id=None) == [104]


def test_local_panel_keeps_intermediate_ancestry_and_marks_projected_edges():
    graph = Graph(pl.DataFrame({"p": [1, 2, 3, 1, 5, 5], "c": [2, 3, 4, 5, 4, 7], "type": ["direct"] * 6}))
    sub = graph.selective_subgraph([1, 3, 4, 7, 99])
    assert set(sub.nodes()) == {1, 3, 4, 7}
    assert {(sub[p], sub[c], rel) for p, c, rel in sub.weighted_edge_list()} == {
        (1, 3, "ancestor_path"), (3, 4, "direct"), (1, 4, "ancestor_path"), (1, 7, "ancestor_path")}


def test_explicit_geometry_for_original_two_tensor_files_and_archived_scale(options, tmp_path):
    legacy = tmp_path / "two-tensor.pt"
    torch.save({"concept_ids": torch.tensor([[1], [2]]), "embeddings": torch.tensor([[0., 0.], [.1, 0.]])}, legacy)
    with pytest.raises(ValueError, match="Declare"):
        load_embedding(legacy)
    assert load_embedding(legacy, geometry="hyperbolic", curvature=2.5)["concept_ids"].shape == (2,)
    model = create_embedding(2, 1, "euclidean", 1., None, 2.)
    with torch.no_grad():
        model.coordinates().copy_(torch.tensor([[1.], [4.]]))
    assert model(torch.tensor([[0, 1]])).item() == 6
    args = replace(options, write_tsv=False, keep_eval_models=False, euclidean_distance_scale=2., threads=2)
    previous_threads = torch.get_num_threads()
    train(args)
    assert torch.get_num_threads() == previous_threads
    folder = output_folder(args, "euclidean")
    assert not list(folder.glob("*.tsv")) and not list((folder / "eval_models").glob("*.pt"))
    assert load_embedding(folder / "embedding.pt")["metadata"]["distance_scale"] == 2.0


def test_recovery_can_finish_a_partially_published_run(options, monkeypatch):
    module = import_module("context.train")
    publish = module.publish_file
    def interrupt(source, destination):
        if destination.name == "embedding.json":
            raise KeyboardInterrupt
        publish(source, destination)
    monkeypatch.setattr(module, "publish_file", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(options)
    monkeypatch.setattr(module, "publish_file", publish)
    monkeypatch.setattr(module, "candidate_loss", lambda *args: pytest.fail("Finished epochs were repeated"))
    train(replace(options, resume=True))
    folder = output_folder(options, options.geometry)
    assert load_embedding(folder / "embedding.pt")["metadata"]["dimensions"] == options.embedding_dim
    assert not list(folder.glob(".context-fit-*"))
    with pytest.raises(ValueError, match="matches"):
        train(replace(options, resume=True, seed=0))


def test_graph_split_is_independent_of_geometry_and_negative_masks_use_only_training_edges(options):
    g = Graph.load(options.graph_file).full_graph
    full = GraphEmbeddingDataset(g, num_negative_samples=400, directed=options.directed)
    args = replace(options, validation_fraction=.12, retrieval_direction="parents")
    a = split_edges(full, args)
    b = split_edges(full, replace(args, geometry="hyperbolic", seed=216, query_seed=args.query_seed))
    np.testing.assert_array_equal(a.validation, b.validation)
    np.testing.assert_array_equal(a.validation_queries, b.validation_queries)
    ds = full.with_training_edges(a.training)
    seen = {tuple(pair) for pair in a.training.tolist()}
    sources, candidates = ds.candidate_nodes(np.arange(ds.training_size), np.random.default_rng(2))
    for (source,), row in zip(sources.tolist(), candidates.tolist()):
        assert (source, row[0]) in seen
        assert set(row[1:]) == set(range(ds.num_nodes)) - {source} - {v for s, v in seen if s == source}


def test_rank_counts_for_high_degree_tied_targets_across_candidate_blocks():
    points = (np.arange(121) % 83).astype(np.float64)[:, None]
    edges = [(0, target) for target in range(1, 89)]
    result = rank_neighbors(points, edge_index(edges, len(points)), [0], candidate_block=11)
    row = np.abs(points[:, 0] - points[0, 0])
    positives = np.arange(1, 89)
    negatives = row[89:]
    ranks = [1 + (negatives < row[p]).sum() + .5 * (negatives == row[p]).sum() for p in positives]
    ap = [(row[positives] <= row[p]).sum() / (row[1:] <= row[p]).sum() for p in positives]
    assert result["mean_rank"] == pytest.approx(np.mean(ranks))
    assert result["macro_average_precision"] == pytest.approx(np.mean(ap))
