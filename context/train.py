"""Train either geometry on a shared concept graph and sampled-neighbor objective."""
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
import polars as pl
import torch
from filelock import FileLock
from hypll.optim import RiemannianAdam, RiemannianSGD

from context.args import Args
from context.dataset import GraphEmbeddingDataset
from context.eval import evaluate_model
from context.export import concept_labels, export_embedding, load_model, save_checkpoint
from context.graph import Graph
from context.loss import candidate_loss
from context.model import create_embedding
from context.stopper import EarlyStopper


def output_folder(args, geometry):
    folder = Path(args.output_directory) / args.experiment_id
    return folder if geometry == "hyperbolic" else folder / "euclidean"


def build_graph(args):
    ids = pl.read_csv(args.concept_id_file, columns=["conceptId"])["conceptId"].to_list()
    if args.relationship_file_path:
        if args.concept_map is None or args.root_node_label is None:
            raise ValueError("SNOMED Is a extraction requires a concept_map and root_node_label")
        return Graph.from_snomed_taxonomy(args.concept_map, args.relationship_file_path, ids,
                                          args.root_node_label, args.graph_id)
    return Graph.from_omop(args.hierarchy_file_path, ids, name=args.graph_id)


def _validate(args):
    if args.geometry not in ("euclidean", "hyperbolic", "both"):
        raise ValueError(f"Unknown geometry {args.geometry!r}")
    if args.optimizer not in ("sgd", "adam"):
        raise ValueError(f"Unknown optimizer {args.optimizer!r}")
    if min(args.epochs, args.save_top, args.batch_size, args.embedding_dim, args.negative_samples) < 1:
        raise ValueError("epochs, save_top, batch_size, embedding_dim and negative_samples must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if args.burn_in and (args.burn_in_epochs < 0 or not math.isfinite(args.burn_in_lr_divisor)
                         or args.burn_in_lr_divisor <= 0):
        raise ValueError("Invalid burn-in schedule")
    if args.eval_interval < 0 or args.lr_reduce_patience < 0 or args.early_stop_patience < 1:
        raise ValueError("Invalid evaluation interval or patience")
    if not 0 < args.lr_reduce_factor < 1:
        raise ValueError("lr_reduce_factor must lie between zero and one")
    if not args.model_file or Path(args.model_file).name != args.model_file:
        raise ValueError("model_file must be a filename")


def _args_with_defaults(args):
    # Older callers pass a Namespace, sometimes with graph_id and concept_map as
    # class attributes. Read attributes to preserve these configurations.
    values = {name: getattr(args, name, default) for name, default in vars(Args()).items()}
    return Args(**{k: str(v) if isinstance(v, Path) else v for k, v in values.items()})


def geometry_settings(args, geometry):
    filename = args.model_file
    if geometry == "euclidean" and filename == Args.model_file:
        filename = "euclidean_embeddings_snomed.pt"
    return replace(args, geometry=geometry, model_file=filename)


def train(args):
    args = _args_with_defaults(args)
    _validate(args)
    folder = Path(args.output_directory) / args.experiment_id
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(folder / ".context.lock"):
        graph_path = Path(args.graph_file) if args.graph_file else folder / "graph.pkl"
        if args.graph_file or graph_path.exists():
            graph = Graph.load(graph_path)
        else:
            graph = build_graph(args)
            with TemporaryDirectory(dir=folder) as temporary:
                path = Path(temporary) / "graph.pkl"
                graph.save(path)
                path.replace(graph_path)
        with graph_path.open("rb") as stream:
            graph_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        subgraph = graph.full_graph if args.graph_id == "full" else graph.subgraph[args.graph_id]
        dataset = GraphEmbeddingDataset(subgraph, directed=args.directed, num_negative_samples=args.negative_samples)
        if not dataset.training_size:
            raise ValueError("No source with a positive edge has an eligible negative neighbor")
        root = None
        if args.root_node_label is not None:
            if args.root_node_label not in dataset.concept_ids:
                raise ValueError(f"Root {args.root_node_label!r} is absent from graph {args.graph_id!r}")
            root = dataset.concept_ids.index(args.root_node_label)
        labels = concept_labels(dataset.concept_ids, args.concept_map)
        geometries = ("hyperbolic", "euclidean") if args.geometry == "both" else (args.geometry,)
        records = []
        for geometry in geometries:
            settings = geometry_settings(args, geometry)
            destination = output_folder(settings, geometry)
            destination.mkdir(exist_ok=True)
            stage = Path(mkdtemp(prefix=".context-fit-", dir=destination))
            try:
                records.extend(_fit(settings, dataset, root, labels, stage, graph_digest))
                _publish(stage, destination, settings.model_file)
            except BaseException:
                if any((stage / "models").glob("*.pt")):
                    print(f"Partial checkpoints retained in {stage}", flush=True)
                else:
                    shutil.rmtree(stage)
                raise
            shutil.rmtree(stage)
        return records


def _fit(args, dataset, root, labels, stage, graph_digest):
    # Independent RNG streams keep paired runs on identical positives and negatives.
    order = torch.Generator().manual_seed(args.seed)
    negatives = np.random.default_rng(args.seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        model = create_embedding(dataset.num_nodes, args.embedding_dim, args.geometry, args.curvature, root)
    model = model.to(args.device)
    forward = torch.compile(model) if args.compile else model
    if args.geometry == "hyperbolic":
        optimizer_class = {"sgd": RiemannianSGD, "adam": RiemannianAdam}[args.optimizer]
    else:
        optimizer_class = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[args.optimizer]

    def run_epoch(optimizer):
        total = torch.zeros((), device=args.device)
        for indices in torch.randperm(dataset.training_size, generator=order).split(args.batch_size):
            sources, targets = dataset.candidate_nodes(indices.numpy(), negatives)
            optimizer.zero_grad(set_to_none=True)
            loss = candidate_loss(forward(sources.to(args.device), targets.to(args.device)))
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(indices)
        mean = total.item() / dataset.training_size
        if not math.isfinite(mean):
            raise FloatingPointError("Non-finite embedding loss")
        return mean

    if args.burn_in and args.burn_in_epochs:
        optimizer = optimizer_class(model.parameters(), lr=args.learning_rate / args.burn_in_lr_divisor)
        for epoch in range(1, args.burn_in_epochs + 1):
            print(f"{args.geometry} burn-in {epoch}: {run_epoch(optimizer):.6f}", flush=True)

    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_reduce_factor, patience=args.lr_reduce_patience)
    stopper = EarlyStopper(patience=args.early_stop_patience)
    losses, records, kept = [], [], []
    (stage / "models").mkdir()
    (stage / "eval_models").mkdir()
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(optimizer)
        losses.append(loss)
        save_model(model, args, stage, epoch, loss, losses, kept=kept)

        if args.eval_interval and epoch % args.eval_interval == 0:
            rank, ap = evaluate_model(model, dataset)
            records.append({**vars(args), "curvature": None if model.curvature is None else model.curvature.item(),
                            "epoch": epoch, "mean_rank": rank, "map_score": ap, "loss": loss})
            save_single_model(model, args, stage, epoch, loss, losses)
        print(f"{args.geometry} epoch {epoch}: {loss:.6f}", flush=True)
        scheduler.step(loss)
        if stopper.update(loss):
            break

    _, best_epoch, best_path = kept[0]
    selected = load_model(best_path)
    export_embedding(stage, selected, dataset.concept_ids, labels, best_epoch)
    summary = {"settings": vars(args), "graph_sha256": graph_digest, "selected_epoch": best_epoch,
               "training_loss": losses, "nodes": dataset.num_nodes, "graph_edges": dataset.num_edges,
               "training_positives": dataset.training_size, "positives_without_negatives": dataset.skipped_positives,
               "optimizer": optimizer_class.__name__}
    (stage / "training.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if records:
        pl.DataFrame(records).write_csv(stage / "eval_metrics.csv")
    return records


def _checkpoint_path(folder, args, epoch, loss):
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"epoch_{epoch}-loss_{loss:.6f}-{args.model_file}"


def save_model(model, args, experiment_folder, epoch, average_loss, loss_per_epoch, *, kept=None):
    """Keep the best training checkpoints. The trainer reuses its in-memory ranking."""
    folder = Path(experiment_folder) / "models"
    if args.save_top < 1:
        raise ValueError("save_top must be positive")
    if kept is None:
        kept = []
        for path in folder.glob("epoch_*-loss_*-*"):
            prefix, _, suffix = path.name.partition("-loss_")
            loss, _, name = suffix.partition("-")
            if name == args.model_file:
                kept.append((float(loss), int(prefix.removeprefix("epoch_")), path))
        kept.sort()
    path = _checkpoint_path(folder, args, epoch, average_loss)
    kept[:] = [entry for entry in kept if entry[2] != path]
    if len(kept) >= args.save_top and average_loss >= kept[-1][0]:
        return
    save_checkpoint(path, model, args, epoch, loss_per_epoch)
    kept.append((float(average_loss), epoch, path))
    kept.sort()
    while len(kept) > args.save_top:
        kept.pop()[2].unlink()


def save_single_model(model, args, experiment_folder, epoch, average_loss, loss_per_epoch):
    path = _checkpoint_path(Path(experiment_folder) / "eval_models", args, epoch, average_loss)
    save_checkpoint(path, model, args, epoch, loss_per_epoch)


def _publish(stage, destination, model_file):
    # Compute the whole fit in staging. A failed fit leaves the last completed
    # export in place. Publish the self-contained embedding last, atomically.
    for name in ("models", "eval_models"):
        target = destination / name
        target.mkdir(exist_ok=True)
        files = list((stage / name).iterdir())
        names = {p.name for p in files}
        for path in files:
            path.replace(target / path.name)
        for old in target.glob("epoch_*"):
            if old.name.endswith("-" + model_file) and old.name not in names:
                old.unlink()
    for name in ("eval_metrics.csv", "labels.tsv", "vec.tsv", "training.json", "embedding.pt"):
        path = stage / name
        if path.exists():
            path.replace(destination / name)
        elif name == "eval_metrics.csv":
            (destination / name).unlink(missing_ok=True)
