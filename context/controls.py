import argparse
import json
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock
import numpy as np
import polars as pl
import torch

from context.dataset import GraphEdges
from context.export import export_embedding, file_sha, load_embedding
from context.graph import Graph
from context.model import create_embedding


def _degree_groups(source, graph_file, ids, metadata):
    if graph_file is None or file_sha(graph_file) != metadata.get("graph_sha256"):
        raise ValueError("Degree matching requires the same graph file that generated the source embedding")
    summary = json.loads((source.parent / "training.json").read_text())
    graph = Graph.load(graph_file)
    name = summary["settings"]["graph_id"]
    dataset = GraphEdges(graph.full_graph if name == "full" else graph.subgraph[name], directed=True)
    if dataset.concept_ids != ids:
        raise ValueError("Graph and source embedding have different concept rows")
    edges = dataset.edges_list.numpy()
    if metadata["training_scope"] == "held_out_edge_selection":
        path = source.parent / "partitions.npz"
        if file_sha(path) != metadata["split_sha256"]:
            raise ValueError("Source edge partition checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            edges = saved["train"]
    incoming = np.bincount(edges[:, 1], minlength=len(ids))
    outgoing_bin = np.floor(np.log2(1 + np.bincount(edges[:, 0], minlength=len(ids)))).astype(np.int64)
    return incoming * (outgoing_bin.max() + 1) + outgoing_bin


def permute_embedding(source, destination, *, seed=1, matching="none", graph_file=None, write_tsv=False):
    source, destination = Path(source), Path(destination)
    source_sha = file_sha(source)
    payload = load_embedding(source)
    meta = payload["metadata"]
    ids = payload["concept_ids"]
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
    if matching not in ("none", "degree"):
        raise ValueError("matching must be none or degree")
    groups = _degree_groups(source, graph_file, ids, meta) if matching == "degree" else np.zeros(len(ids), dtype=np.int64)
    fixed = [row for row in (meta.get("root_index"), meta.get("graph_root_index")) if row is not None]
    eligible = np.flatnonzero(~np.isin(np.arange(len(ids)), fixed))
    order = eligible[np.argsort(groups[eligible], kind="stable")]
    cuts = np.flatnonzero(np.diff(groups[order])) + 1
    permutation = np.arange(len(ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for cluster in np.split(order, cuts):
        shuffled = rng.permutation(cluster)
        permutation[shuffled] = np.roll(shuffled, 1)
    points = payload["embeddings"]
    with torch.random.fork_rng(devices=[]):
        model = create_embedding(len(ids), points.shape[1], meta["geometry"], meta["effective_curvature_magnitude"],
                                 meta.get("root_index"), meta.get("distance_scale", 1.)).to(dtype=points.dtype)
    different = 0
    with torch.no_grad():
        for start in range(0, len(ids), 32768):
            block = slice(start, start + 32768)
            shuffled = points[torch.from_numpy(permutation[block])]
            different += int(torch.any(points[block] != shuffled, dim=1).sum())
            model.coordinates()[block].copy_(shuffled)
    labels = ids
    if write_tsv and (source.parent / "labels.tsv").is_file():
        labels = pl.read_csv(source.parent / "labels.tsv", has_header=False, separator="\t", infer_schema=False).to_series().to_list()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(destination.parent / ("." + destination.name + ".lock")):
        if destination.exists():
            raise FileExistsError(f"Choose a new destination for the permutation: {destination}")
        stage = Path(tempfile.mkdtemp(prefix=".context-permutation-", dir=destination.parent))
        try:
            np.save(stage / "permutation.npy", permutation, allow_pickle=False)
            if file_sha(source) != source_sha:
                raise ValueError("Source embedding changed during permutation")
            info = {"training_method": "permutation_control", "source_method": meta.get("training_method"),
                    "source_sha256": source_sha, "graph_sha256": meta.get("graph_sha256"),
                    "source_training_scope": meta.get("training_scope"), "split_sha256": meta.get("split_sha256"),
                    "matching": matching, "seed": seed, "rows_reassigned": int(np.count_nonzero(permutation != np.arange(len(ids)))),
                    "rows_with_changed_vector": different, "permutation_sha256": file_sha(stage / "permutation.npy")}
            export_embedding(stage, model, ids, labels, meta.get("selected_epoch"), graph_root_index=meta.get("graph_root_index"),
                             provenance=info, write_tsv=write_tsv)
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return info


def main():
    parser = argparse.ArgumentParser(description="Create a control embedding by shuffling concept-to-vector assignments.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--matching", choices=("none", "degree"), default="none")
    parser.add_argument("--graph-file", type=Path)
    parser.add_argument("--write-tsv", action="store_true")
    print(json.dumps(permute_embedding(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
