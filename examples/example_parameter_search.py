import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import simple_parsing

from context.args import Args
from context.train import build_graph, geometry_settings, output_folder, train


def cached_metrics(settings, graph_digest):
    folder = output_folder(settings, settings.geometry)
    summary = folder / "training.json"
    if not summary.exists() or not (folder / "embedding.pt").exists():
        return None
    info = json.loads(summary.read_text())
    info["settings"].setdefault("euclidean_root", "fixed")
    if info["graph_sha256"] != graph_digest or info["settings"] != vars(settings):
        return None
    metrics = folder / "eval_metrics.csv"
    return pl.read_csv(metrics).to_dicts() if metrics.exists() else []


def main():
    args = simple_parsing.parse(Args)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    graph_path = Path(args.graph_file) if args.graph_file else output / "graph.pkl"
    if not args.graph_file and not graph_path.exists():
        with TemporaryDirectory(dir=output) as tmp:
            path = Path(tmp) / "graph.pkl"
            build_graph(args).save(path)
            path.replace(graph_path)
    with graph_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    geometries = ("hyperbolic", "euclidean") if args.geometry == "both" else (args.geometry,)
    records = []
    for i, (directed, negatives, dim, burn) in enumerate(product(
            (True, False), (10, 50, 100), (3, 10, 30, 100), (10, 100))):
        for geometry in geometries:
            settings = replace(args, graph_file=str(graph_path), experiment_id=f"{args.experiment_id}_{i:03d}",
                               geometry=geometry, directed=directed, negative_samples=negatives,
                               embedding_dim=dim, burn_in_epochs=burn)
            settings = geometry_settings(settings, geometry)
            cached = cached_metrics(settings, digest)
            records.extend(train(settings) if cached is None else cached)
        if records:
            pl.DataFrame(records, infer_schema_length=None).write_csv(output / "eval_metrics_main.csv")


if __name__ == "__main__":
    main()
