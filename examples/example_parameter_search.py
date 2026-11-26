import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path
from filelock import FileLock

import polars as pl
import simple_parsing

from context.args import Args
from context.export import file_sha
from context.recovery import completed_records, fit_identity
from context.train import graph_for_run, geometry_settings, output_folder, train


def cached_metrics(settings, graph_digest):
    folder = output_folder(settings, settings.geometry)
    if not (folder / "training.json").is_file():
        return None
    labels = file_sha(settings.concept_map) if settings.concept_map else None
    identity = fit_identity(settings, graph_digest, None, {"selection": None, "labels": labels})
    return completed_records(folder, identity)


def main():
    args = simple_parsing.parse(Args)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    graph_path = Path(args.graph_file) if args.graph_file else output / "graph.pkl"
    if not args.graph_file:
        with FileLock(output / ".context-graph.lock"):
            graph_for_run(args, graph_path)
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
