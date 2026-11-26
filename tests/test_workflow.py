import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from context.args import Args
from context.export import load_model
from context.graph import Graph
from context.train import train


@pytest.fixture
def settings(tmp_path):
    edges = [(33, 11), (90, 33), (90, 27), (27, 11), (27, 82), (33, 45)]
    graph = Graph(pl.DataFrame([(a, b, "Is a") for a, b in edges],
                               schema=["source", "target", "edge"], orient="row"))
    graph.ancestral_subgraph([11, 82, 45])
    graph_file = tmp_path / "shared graph.pkl"
    graph.save(graph_file)
    return Args(output_directory=str(tmp_path), experiment_id="trial", graph_file=str(graph_file),
                root_node_label=90, geometry="both", seed=7, epochs=3, learning_rate=.02,
                batch_size=5, burn_in=True, burn_in_epochs=1, eval_interval=1, save_top=2,
                curvature=2.5, negative_samples=3, embedding_dim=2)


def assert_exports(args):
    base = Path(args.output_directory) / args.experiment_id
    ids = Graph.load(args.graph_file).subgraph["ancestral"].nodes()
    for geometry, folder in [("hyperbolic", base), ("euclidean", base / "euclidean")]:
        data = torch.load(folder / "embedding.pt", weights_only=True)
        assert data["concept_ids"].tolist() == list(ids)
        assert data["geometry"] == geometry
        assert data["curvature"] == (2.5 if geometry == "hyperbolic" else None)
        assert ids[data["root_index"]] == 90 and torch.count_nonzero(data["embeddings"][data["root_index"]]) == 0
        assert torch.count_nonzero(data["embeddings"]) > 0
        np.testing.assert_allclose(np.loadtxt(folder / "vec.tsv"), data["embeddings"].numpy(), rtol=1e-6)
        assert [int(row[0]) for row in csv.reader((folder / "labels.tsv").read_text().splitlines())] == list(ids)
        checkpoints = list((folder / "models").glob("*.pt"))
        assert len(checkpoints) == min(args.epochs, args.save_top)
        checkpoint = next(p for p in checkpoints if p.name.startswith(f"epoch_{data['epoch']}-"))
        restored = load_model(checkpoint)
        torch.testing.assert_close(restored.coordinates(), data["embeddings"], rtol=0, atol=0)
        training = json.loads((folder / "training.json").read_text())
        saved = torch.load(checkpoint, weights_only=True)
        np.testing.assert_allclose(saved["losses"].numpy(), training["training_loss"][:saved["epoch"]])
        assert saved["epoch"] == np.argmin(training["training_loss"]) + 1


def test_paired_run_matches_standalone_and_roundtrips(settings):
    records = train(settings)
    assert len(records) == 2 * settings.epochs
    assert {r["geometry"] for r in records} == {"euclidean", "hyperbolic"}
    assert_exports(settings)
    single = replace(settings, geometry="euclidean", experiment_id="single")
    rng_state = torch.random.get_rng_state()
    assert [r["loss"] for r in train(single)] == [r["loss"] for r in records if r["geometry"] == "euclidean"]
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    base = Path(settings.output_directory)
    first = torch.load(base / "trial/euclidean/embedding.pt", weights_only=True)
    second = torch.load(base / "single/euclidean/embedding.pt", weights_only=True)
    assert torch.equal(first["embeddings"], second["embeddings"])


def test_legacy_namespace_and_failed_rerun(settings):
    class Legacy:
        graph_id = "ancestral"
        root_node_label = 90

    legacy = Legacy()
    for k, v in vars(settings).items():
        if k not in ("geometry", "graph_id", "root_node_label"):
            setattr(legacy, k, v)
    assert {r["geometry"] for r in train(legacy)} == {"hyperbolic"}
    output = Path(settings.output_directory) / settings.experiment_id / "embedding.pt"
    original = output.read_bytes()
    with pytest.raises(Exception, match="missing"):
        train(replace(settings, geometry="hyperbolic", concept_map="missing.tsv"))
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="Root"):
        train(replace(settings, root_node_label=-1))
    train(replace(settings, geometry="hyperbolic", epochs=1, save_top=1, eval_interval=0))
    assert len(list((output.parent / "models").glob("*.pt"))) == 1
    assert not list((output.parent / "eval_models").glob("*.pt"))
    assert not (output.parent / "eval_metrics.csv").exists()


def test_cli_both_from_a_vocabulary_table(tmp_path):
    ancestors = tmp_path / "ancestor.tsv"
    ancestors.write_text("ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\tmax_levels_of_separation\n"
                         "10\t20\t1\t1\n10\t40\t1\t2\n20\t40\t1\t1\n10\t30\t1\t1\n30\t50\t1\t1\n10\t50\t2\t2\n")
    (tmp_path / "ids.csv").write_text("conceptId\n40\n50\n")
    names = tmp_path / "concept.tsv"
    names.write_text("concept_id\tconcept_name\n10\tRoot\n20\tFirst\n30\tSecond\n40\tLeaf A\n50\tLeaf B\n")
    subprocess.run([sys.executable, "-m", "context", "--geometry", "both", "--output_directory", str(tmp_path),
                    "--experiment_id", "cli", "--root_node_label", "10", "--hierarchy_file_path", str(ancestors),
                    "--concept_id_file", str(tmp_path / "ids.csv"), "--concept_map", str(names), "--epochs", "1",
                    "--burn_in", "false", "--eval_interval", "1", "--batch_size", "3",
                    "--negative_samples", "2", "--learning_rate", ".01"], check=True, cwd=tmp_path)
    folder = tmp_path / "cli"
    graph = Graph.load(folder / "graph.pkl").subgraph["ancestral"]
    assert (10, 40) in {(graph[s], graph[t]) for s, t in graph.edge_list()}
    assert (10, 50) not in {(graph[s], graph[t]) for s, t in graph.edge_list()}
    for directory in (folder, folder / "euclidean"):
        data = torch.load(directory / "embedding.pt", weights_only=True)
        labels = (directory / "labels.tsv").read_text().splitlines()
        assert labels[data["concept_ids"].tolist().index(40)] == "Leaf A"


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_accelerator_paired_training(settings, device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    args = replace(settings, device=device, optimizer="adam", epochs=1)
    train(args)
    assert_exports(args)


def test_interrupted_update_keeps_last_export_and_partial_checkpoint(settings, monkeypatch):
    import importlib
    module = importlib.import_module("context.train")
    args = replace(settings, geometry="euclidean", epochs=1)
    train(args)
    folder = Path(args.output_directory) / args.experiment_id / "euclidean"
    original = (folder / "embedding.pt").read_bytes()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "evaluate_model", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(args)
    assert (folder / "embedding.pt").read_bytes() == original
    partial, = folder.glob(".context-fit-*/models/*.pt")
    assert load_model(partial).geometry == "euclidean"


def test_parameter_search_reuses_only_matching_runs(settings, monkeypatch):
    import examples.example_parameter_search as grid
    import examples.example_plot as plot

    plot.plt.switch_backend("Agg")
    monkeypatch.setattr(grid.simple_parsing, "parse", lambda cls: replace(settings, epochs=1))
    monkeypatch.setattr(grid, "product", lambda *values: [(True, 2, 2, 0)])
    grid.main()
    output = Path(settings.output_directory)
    rows = pl.read_csv(output / "eval_metrics_main.csv")
    assert set(rows["geometry"]) == {"hyperbolic", "euclidean"}
    for geom, group in rows.partition_by("geometry", as_dict=True).items():
        plot.plot_geometry(group, geom[0])
    plot.plt.close("all")

    def unexpected(*args):
        pytest.fail("Completed grid configuration was fitted twice")

    monkeypatch.setattr(grid, "train", unexpected)
    grid.main()
    changed = replace(settings, epochs=4, experiment_id="trial_000")
    assert grid.cached_metrics(changed, "changed graph") is None


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_trainable_root_with_typed_graph_and_paired_cli(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    ids = tmp_path / "seeds.csv"
    ids.write_text("conceptId\n11\n45\n82\n")
    concepts = tmp_path / "concepts.tsv"
    concepts.write_text("concept_id\tconcept_name\tvocabulary_id\tstandard_concept\tinvalid_reason\n" +
                        "".join(f"{i}\tconcept {i}\tSNOMED\tS\t\n" for i in [11, 27, 33, 45, 82, 90]))
    links = tmp_path / "relationships.tsv"
    links.write_text("concept_id_1\tconcept_id_2\trelationship_id\tinvalid_reason\n" +
                     "".join(f"{child}\t{parent}\tIs a\t\n" for child, parent in
                             [(33, 90), (27, 90), (11, 33), (11, 27), (45, 33), (82, 27)]))
    cmd = [sys.executable, "-m", "context", "--output_directory", str(tmp_path), "--experiment_id", "roots",
           "--root_node_label", "90", "--concept_id_file", str(ids), "--concept_map", str(concepts),
           "--relationship_file_path", str(links), "--epochs", "2", "--burn_in", "false", "--eval_interval", "0",
           "--optimizer", "adam", "--learning_rate", ".02", "--negative_samples", "3", "--device", device]
    subprocess.run(cmd + ["--geometry", "both", "--euclidean_root", "trainable"], check=True, cwd=tmp_path)
    folder = tmp_path / "roots"
    hyper = torch.load(folder / "embedding.pt", weights_only=True)
    separate = folder / "euclidean" / "trainable-root"
    free = torch.load(separate / "embedding.pt", weights_only=True)
    root = free["graph_root_index"]
    assert torch.equal(hyper["concept_ids"], free["concept_ids"])
    assert free["concept_ids"][root] == 90
    assert root == hyper["graph_root_index"] == hyper["root_index"]
    assert free["root_index"] is None
    assert torch.count_nonzero(hyper["embeddings"][root]) == 0
    assert torch.count_nonzero(free["embeddings"][root]) > 0
    checkpoint = next((separate / "models").glob(f"epoch_{free['epoch']}-*.pt"))
    restored = load_model(checkpoint)
    assert restored.root_concept_index is None
    torch.testing.assert_close(restored.coordinates(), free["embeddings"], rtol=0, atol=0)
    assert torch.load(checkpoint, weights_only=True)["args"]["root_node_label"] == 90

    subprocess.run(cmd + ["--geometry", "euclidean"], check=True, cwd=tmp_path)
    fixed = torch.load(folder / "euclidean" / "embedding.pt", weights_only=True)
    assert fixed["graph_root_index"] == fixed["root_index"] == root
    assert torch.count_nonzero(fixed["embeddings"][root]) == 0
    repeated = torch.load(separate / "embedding.pt", weights_only=True)
    assert torch.equal(repeated["embeddings"], free["embeddings"])
