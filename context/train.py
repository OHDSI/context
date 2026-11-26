import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
from filelock import FileLock

from context.args import Args
from context.classical import METHODS, fit_classical, validate_settings
from context.dataset import GraphEdges, GraphEmbeddingDataset
from context.eval import evaluate_model
from context.export import concept_labels, export_embedding, file_sha, load_model, save_checkpoint
from context.graph import Graph
from context.loss import candidate_loss
from context.model import create_embedding
from context.stopper import EarlyStopper
from context.optim import make_optimizer
from context.recovery import completed_records, fit_identity, matching_stage, new_stage, publish_file, recover, save_progress
from context.selection import split_edges


def output_folder(args, geometry):
    folder = Path(args.output_directory) / args.experiment_id
    if getattr(args, "training_method", "sampled") != "sampled":
        folder /= args.training_method
    if geometry == "euclidean":
        folder = folder / "euclidean"
        if getattr(args, "euclidean_root", "fixed") == "trainable":
            folder = folder / "trainable-root"
    return folder


def build_graph(args):
    if args.taxonomy_scope != "seeds" and not args.relationship_file_path:
        raise ValueError("Whole-taxonomy extraction requires a typed SNOMED relationship file")
    ids = None if args.taxonomy_scope != "seeds" else pl.read_csv(args.concept_id_file, columns=["conceptId"])["conceptId"]
    if args.relationship_file_path:
        if args.concept_map is None or (args.root_node_label is None and args.taxonomy_scope != "all"):
            raise ValueError("SNOMED Is a extraction requires a concept_map and root_node_label")
        return Graph.from_snomed_taxonomy(args.concept_map, args.relationship_file_path, ids,
                                          args.root_node_label, args.graph_id, include_all=args.taxonomy_scope == "all",
                                          snomed_concepts=args.snomed_concepts)
    return Graph.from_omop(args.hierarchy_file_path, ids, name=args.graph_id)


def _validate(args):
    if args.geometry not in ("euclidean", "hyperbolic", "both"):
        raise ValueError(f"Unknown geometry {args.geometry!r}")
    if args.euclidean_root not in ("fixed", "trainable"):
        raise ValueError("euclidean_root must be fixed or trainable")
    if args.training_method == "landmark":
        if (args.landmarks is not None and args.landmarks < 2) or args.embedding_dim < 1:
            raise ValueError("Landmark fitting requires at least two landmarks and a positive embedding dimension")
        if args.landmark_edge_length is not None and (not math.isfinite(args.landmark_edge_length) or args.landmark_edge_length <= 0):
            raise ValueError("landmark_edge_length must be finite and positive")
        if str(args.device) != "cpu":
            raise ValueError("Landmark shortest paths and spectral fitting run on CPU. Use --device cpu.")
    elif args.training_method == "sampled":
        _validate_sampled(args)
    elif args.training_method in METHODS:
        validate_settings(args)
    else:
        raise ValueError(f"Unknown training_method {args.training_method!r}")
    if args.eval_interval < 0:
        raise ValueError("Invalid evaluation interval")
    if not 0 <= args.validation_fraction < .5 or (args.validation_fraction and not args.eval_interval):
        raise ValueError("Validation and test fractions must be below .5 and require a positive evaluation interval")
    if args.taxonomy_scope not in ("seeds", "subtree", "all"):
        raise ValueError("Unknown taxonomy scope")
    if args.threads is not None and args.threads < 1:
        raise ValueError("threads must be positive")
    if args.split_policy not in ("random", "preserve_degrees"):
        raise ValueError("Unknown edge split policy")
    if args.eval_queries < 0 or args.checkpoint_interval < 1:
        raise ValueError("Invalid query count or checkpoint interval")
    if args.retrieval_direction not in ("observed", "parents", "children"):
        raise ValueError("Unknown retrieval direction")
    if args.sgd_updates not in ("auto", "dense", "sparse"):
        raise ValueError("Unknown SGD update policy")
    if args.retrieval_direction != "observed" and not args.validation_fraction:
        raise ValueError("Explicit parent/child retrieval requires a held-out edge split")
    if not args.model_file or Path(args.model_file).name != args.model_file:
        raise ValueError("model_file must be a filename")


def _validate_sampled(args):
    if args.optimizer not in ("sgd", "adam"):
        raise ValueError(f"Unknown optimizer {args.optimizer!r}")
    if min(args.epochs, args.save_top, args.batch_size, args.embedding_dim, args.negative_samples) < 1:
        raise ValueError("epochs, save_top, batch_size, embedding_dim and negative_samples must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if args.burn_in and (args.burn_in_epochs < 0 or not math.isfinite(args.burn_in_lr_divisor)
                         or args.burn_in_lr_divisor <= 0):
        raise ValueError("Invalid burn-in schedule")
    if args.lr_reduce_patience < 0 or args.early_stop_patience < 1:
        raise ValueError("Invalid evaluation interval or patience")
    if not 0 < args.lr_reduce_factor < 1:
        raise ValueError("lr_reduce_factor must lie between zero and one")


def _args_with_defaults(args):
    values = {name: getattr(args, name, default) for name, default in vars(Args()).items()}
    if hasattr(args, "validation_queries"):
        values["eval_queries"] = args.validation_queries
    return Args(**{k: str(v) if isinstance(v, Path) else v for k, v in values.items()})


def geometry_settings(args, geometry):
    filename = args.model_file
    if geometry == "euclidean" and filename == Args.model_file:
        filename = "euclidean_embeddings_snomed.pt"
    root_mode = args.euclidean_root if geometry == "euclidean" else "fixed"
    return replace(args, geometry=geometry, model_file=filename, euclidean_root=root_mode)


def _source_hashes(args):
    files = ({"concepts": args.concept_map, "relationships": args.relationship_file_path} if args.relationship_file_path
             else {"ancestry": args.hierarchy_file_path})
    if args.taxonomy_scope == "seeds":
        files["seeds"] = args.concept_id_file
    return {"files": {key: file_sha(path) for key, path in files.items()}, "root": args.root_node_label,
            "graph_id": args.graph_id, "taxonomy_scope": args.taxonomy_scope, "snomed_concepts": args.snomed_concepts}


def graph_for_run(args, path):
    if args.graph_file:
        return Graph.load(path)
    if path.exists():
        graph = Graph.load(path)
        if getattr(graph, "source_inputs", None) is None:
            if args.relationship_file_path or Path(args.concept_id_file).is_file():
                raise ValueError("Legacy graph cache has no input hashes. Select it with graph_file or choose a new experiment.")
        elif _source_hashes(args) != graph.source_inputs:
            raise ValueError("Graph inputs changed. Use a fresh experiment ID or explicitly select the saved graph_file.")
        return graph
    inputs = _source_hashes(args)
    graph = build_graph(args)
    graph.source_inputs = inputs
    graph.save(path)
    return graph


def train(args):
    args = _args_with_defaults(args)
    if args.refit_from:
        from context.refit import refit_selected
        return refit_selected(args.refit_from, args)["evaluation_records"]
    return _train(args)


def _train(args, fixed_rates=None, selection_identity=None):
    _validate(args)
    previous_threads = torch.get_num_threads()
    if args.threads is None or args.threads == previous_threads:
        return _train_geometries(args, fixed_rates, selection_identity)
    torch.set_num_threads(args.threads)
    try:
        return _train_geometries(args, fixed_rates, selection_identity)
    finally:
        torch.set_num_threads(previous_threads)


def _train_geometries(args, fixed_rates, selection_identity):
    folder = Path(args.output_directory) / args.experiment_id
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(folder / ".context.lock"):
        graph_path = Path(args.graph_file) if args.graph_file else folder / "graph.pkl"
        graph = graph_for_run(args, graph_path)
        graph_digest = file_sha(graph_path)
        if selection_identity and selection_identity.get("graph_sha256", graph_digest) != graph_digest:
            raise ValueError("Refit requires the graph used for selection")
        subgraph = graph.full_graph if args.graph_id == "full" else graph.subgraph[args.graph_id]
        if args.training_method != "sampled":
            dataset = GraphEdges(subgraph, directed=args.directed)
        else:
            dataset = GraphEmbeddingDataset(subgraph, directed=args.directed, num_negative_samples=args.negative_samples)
        del graph, subgraph
        partition = None
        if args.validation_fraction:
            partition = split_edges(dataset, args)
            dataset = dataset.with_training_edges(partition.training)
        if args.training_method == "sampled" and not dataset.training_size:
            raise ValueError("No source with a positive edge has an eligible negative neighbor")
        root = None
        if args.root_node_label is not None:
            try:
                root = dataset.concept_ids.index(args.root_node_label)
            except ValueError as error:
                raise ValueError(f"Root {args.root_node_label!r} is absent from graph {args.graph_id!r}") from error
        labels = concept_labels(dataset.concept_ids, args.concept_map)
        label_digest = file_sha(args.concept_map) if args.concept_map else None
        geometries = ("hyperbolic", "euclidean") if args.geometry == "both" else (args.geometry,)
        records = []
        basis = None
        for geometry in geometries:
            settings = geometry_settings(args, geometry)
            destination = output_folder(settings, geometry)
            destination.mkdir(parents=True, exist_ok=True)
            identity = fit_identity(settings, graph_digest, fixed_rates, {"selection": selection_identity, "labels": label_digest})
            stage = matching_stage(destination, identity) if args.resume else None
            if args.resume:
                cached = completed_records(stage if stage is not None else destination, identity)
                if cached is not None:
                    if stage is not None:
                        _publish(stage, destination, settings.model_file)
                        shutil.rmtree(stage)
                    records.extend(cached)
                    continue
                if stage is None and ((destination / "training.json").exists() or any(destination.glob(".context-fit-*/progress.pt")) or any(destination.glob(".context-fit-*/walk.model"))):
                    raise ValueError("No saved run matches the graph, settings and implementation")
            if stage is None:
                stage = new_stage(destination, identity)
            try:
                if settings.training_method == "sampled":
                    fitted = _fit(settings, dataset, root, labels, stage, graph_digest, partition, fixed_rates, selection_identity)
                else:
                    if settings.training_method == "landmark":
                        from context.landmarks import fit_coordinates, landmark_distances
                        if basis is None:
                            budget = settings.landmarks or max(64, 2 * settings.embedding_dim + 1)
                            basis = landmark_distances(dataset, root, budget, settings.seed)
                        model, details = fit_coordinates(basis, settings, root)
                    else:
                        model, details = fit_classical(dataset, settings, root, stage)
                    fitted = _export_representation(settings, dataset, model, details, root, labels, stage, graph_digest, partition, selection_identity)
                records.extend(fitted)
                _publish(stage, destination, settings.model_file)
            except BaseException:
                if (stage / "training.json").exists() or (stage / "progress.pt").exists() or (stage / "walk.model").exists() or any((stage / "models").glob("*.pt")):
                    print(f"Partial checkpoints retained in {stage}", flush=True)
                else:
                    shutil.rmtree(stage)
                raise
            shutil.rmtree(stage)
        return records


def _export_representation(args, dataset, model, details, root, labels, stage, graph_digest, partition, selection_identity):
    scope = "held_out_edge_selection" if partition else ("full_graph_refit" if selection_identity else "full_graph_reconstruction")
    provenance = {"training_method": args.training_method, "graph_sha256": graph_digest, "training_scope": scope,
                  "selection_identity": selection_identity, "seed": args.seed, **details}
    records, counts = [], {}
    if partition:
        np.savez(stage / "partitions.npz", train=partition.training, validation=partition.validation, test=partition.test)
        provenance["split_sha256"] = file_sha(stage / "partitions.npz")
        metrics = partition.score(model, "validation")
        counts = {"validation_edges": len(partition.validation), "test_edges": len(partition.test),
                  "best_validation_ap": metrics["macro_average_precision"], "test": partition.score(model, "test"),
                  "nodes_without_training_edges": dataset.num_nodes - len(np.unique(partition.training))}
    elif args.eval_interval and dataset.num_edges:
        rank, ap = evaluate_model(model, dataset, query_limit=args.eval_queries, query_seed=args.query_seed)
        eligible = int(np.count_nonzero(np.diff(dataset.neighbor_rows.indptr)))
        metrics = {"mean_rank": rank, "macro_average_precision": ap, "eligible_queries": eligible,
                   "queries": min(eligible, args.eval_queries or eligible), "candidates": dataset.num_nodes}
    else:
        metrics = None
    if metrics is not None:
        records.append({**vars(args), **metrics, "evaluation_scope": "validation" if partition else "reconstruction",
                        "map_score": metrics["macro_average_precision"], "epoch": None})
        pl.DataFrame(records).write_csv(stage / "eval_metrics.csv")
    export_embedding(stage, model, dataset.concept_ids, labels, None, graph_root_index=root,
                     provenance=provenance, write_tsv=args.write_tsv)
    files = ["embedding.pt", "embedding.json", "vec.tsv", "labels.tsv", "eval_metrics.csv", "partitions.npz"]
    total_edges = dataset.num_edges + (len(partition.validation) + len(partition.test) if partition else 0)
    summary = {"settings": vars(args), **provenance, **counts, "selected_epoch": None,
               "training_loss": [], "learning_rates": [], "nodes": dataset.num_nodes, "graph_edges": total_edges,
               "training_edges": dataset.num_edges, "evaluation_records": records, "fit_identity": (stage / "identity").read_text(),
               "output_sha256": {name: file_sha(stage / name) for name in files if (stage / name).exists()}}
    (stage / "training.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return records


def _fit(args, dataset, root, labels, stage, graph_digest, partition=None, fixed_rates=None, selection_identity=None):
    order = torch.Generator().manual_seed(args.seed)
    negatives = np.random.default_rng(args.seed)
    anchor = None if args.geometry == "euclidean" and args.euclidean_root == "trainable" else root
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        model = create_embedding(dataset.num_nodes, args.embedding_dim, args.geometry, args.curvature, anchor,
                                 args.euclidean_distance_scale)
        if root is not None and anchor is None:
            with torch.no_grad():
                model.coordinates()[root].zero_()
    model = model.to(args.device)
    burn_epochs = args.burn_in_epochs if args.burn_in else 0
    optimizer = make_optimizer(model, args, args.learning_rate / args.burn_in_lr_divisor if burn_epochs else args.learning_rate)
    forward = torch.compile(model) if args.compile else model
    def scheduler_for(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_reduce_factor,
                                                         patience=args.lr_reduce_patience)
    scheduler = scheduler_for(optimizer)
    stopper = EarlyStopper(patience=args.early_stop_patience)
    def run_epoch():
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

    for directory in ("models", "eval_models"):
        (stage / directory).mkdir(exist_ok=True)
    state = {"burn_completed": 0, "losses": [], "rates": [], "records": [], "kept": [],
             "best_validation": -1.0, "best_epoch": None, "done": False}
    if (stage / "progress.pt").exists():
        state = recover(stage, model, optimizer, order, negatives, scheduler, stopper)
        recorded = {record["epoch"]: record["loss"] for record in state["records"]}
        for path in (stage / "eval_models").glob("epoch_*.pt"):
            epoch = int(path.name.split("-")[0].removeprefix("epoch_"))
            if epoch not in recorded or path != _checkpoint_path(path.parent, args, epoch, recorded[epoch]):
                path.unlink()
    losses, records = state["losses"], state["records"]
    kept = [(score, epoch, stage / "models" / name) for score, epoch, name in state["kept"]]
    def checkpoint():
        state["kept"] = [(score, epoch, path.name) for score, epoch, path in kept]
        save_progress(stage, model, optimizer, order, negatives, scheduler, stopper, state)
        keep_paths = {entry[2] for entry in kept}
        if not args.keep_eval_models:
            selected_epoch = state["best_epoch"]
            for path in (stage / "eval_models").glob("epoch_*.pt"):
                if not path.name.startswith(f"epoch_{selected_epoch}-"):
                    path.unlink()
        for path in (stage / "models").glob("epoch_*.pt"):
            if path not in keep_paths:
                path.unlink()
    for epoch in range(state["burn_completed"] + 1, burn_epochs + 1):
        print(f"{args.geometry} burn-in {epoch}: {run_epoch():.6f}", flush=True)
        state["burn_completed"] = epoch
        if epoch == 1 or epoch == burn_epochs or epoch % args.checkpoint_interval == 0:
            checkpoint()
    if burn_epochs and not losses:
        optimizer = make_optimizer(model, args, args.learning_rate)
        scheduler = scheduler_for(optimizer)
    curvature = None if model.curvature is None else model.curvature.item()
    eligible_queries = int(np.count_nonzero(np.diff(dataset.neighbor_rows.indptr)))
    for epoch in range(len(losses) + 1, args.epochs + 1):
        if state["done"]:
            break
        if fixed_rates:
            optimizer.param_groups[0]["lr"] = fixed_rates[epoch - 1]
        state["rates"].append(optimizer.param_groups[0]["lr"])
        loss = run_epoch()
        losses.append(loss)
        save_model(model, args, stage, epoch, loss, losses, kept=kept, prune=False)
        evaluate_now = args.eval_interval and (epoch % args.eval_interval == 0 or
                                               (partition is not None and epoch in (1, args.epochs)))
        if evaluate_now:
            if partition is None:
                rank, ap = evaluate_model(model, dataset, query_limit=args.eval_queries, query_seed=args.query_seed)
                metrics = {"mean_rank": rank, "macro_average_precision": ap, "eligible_queries": eligible_queries,
                           "queries": min(eligible_queries, args.eval_queries or eligible_queries),
                           "candidates": dataset.num_nodes}
            else:
                metrics = partition.score(model, "validation")
                if metrics["macro_average_precision"] > state["best_validation"]:
                    state["best_validation"] = metrics["macro_average_precision"]
                    state["best_epoch"] = epoch
            records.append({**vars(args), **metrics, "curvature": curvature, "epoch": epoch,
                            "evaluation_scope": "validation" if partition else "reconstruction",
                            "map_score": metrics["macro_average_precision"], "loss": loss})
            if args.keep_eval_models or (partition and state["best_epoch"] == epoch):
                save_single_model(model, args, stage, epoch, loss, losses)
        if not fixed_rates and (partition is None or evaluate_now):
            criterion = -metrics["macro_average_precision"] if partition else loss
            scheduler.step(criterion)
            state["done"] = stopper.update(criterion)
        state["done"] = state["done"] or epoch == args.epochs
        print(f"{args.geometry} epoch {epoch}: {loss:.6f}", flush=True)
        if epoch == 1 or state["done"] or epoch % args.checkpoint_interval == 0:
            checkpoint()

    best_epoch = state["best_epoch"] if partition else (len(losses) if fixed_rates else kept[0][1])
    if partition:
        best_path = _checkpoint_path(stage / "eval_models", args, best_epoch, losses[best_epoch - 1])
    elif fixed_rates and best_epoch not in [entry[1] for entry in kept]:
        best_path = _checkpoint_path(stage / "models", args, best_epoch, losses[-1])
        save_checkpoint(best_path, model, args, best_epoch, losses)
    else:
        best_path = next(path for _, epoch, path in kept if epoch == best_epoch)
    selected = load_model(best_path)
    mode = "held_out_edge_selection" if partition else ("full_graph_refit" if fixed_rates else "full_graph_reconstruction")
    provenance = {"training_method": "sampled", "graph_sha256": graph_digest, "training_scope": mode, "seed": args.seed,
                  "selection_identity": selection_identity}
    split_counts = {}
    if partition:
        np.savez(stage / "partitions.npz", train=partition.training, validation=partition.validation, test=partition.test)
        provenance["split_sha256"] = file_sha(stage / "partitions.npz")
        split_counts = {"training_edges": len(partition.training), "validation_edges": len(partition.validation),
                        "test_edges": len(partition.test), "best_validation_ap": state["best_validation"],
                        "test": partition.score(selected, "test"),
                        "nodes_without_training_edges": dataset.num_nodes - len(np.unique(partition.training))}
    embedding_hash = export_embedding(stage, selected, dataset.concept_ids, labels, best_epoch,
                                      graph_root_index=root, provenance=provenance, write_tsv=args.write_tsv)
    if records:
        pl.DataFrame(records, infer_schema_length=None).write_csv(stage / "eval_metrics.csv")
    files = ["embedding.json", str(best_path.relative_to(stage))]
    if args.write_tsv:
        files.extend(["vec.tsv", "labels.tsv"])
    if partition:
        files.append("partitions.npz")
    if records:
        files.append("eval_metrics.csv")
    summary = {"settings": vars(args), **provenance, **split_counts, "selected_epoch": best_epoch,
               "training_loss": losses, "learning_rates": state["rates"], "nodes": dataset.num_nodes,
               "graph_edges": dataset.num_edges if not partition else sum(len(p) for p in
                              (partition.training, partition.validation, partition.test)),
               "training_edges": dataset.num_edges, "training_positives": dataset.training_size,
               "positives_without_negatives": dataset.skipped_positives, "optimizer": type(optimizer).__name__,
               "evaluation_records": records, "fit_identity": (stage / "identity").read_text(),
               "output_sha256": {"embedding.pt": embedding_hash, **{name: file_sha(stage / name) for name in files}}}
    (stage / "training.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return records


def _checkpoint_path(folder, args, epoch, loss):
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"epoch_{epoch}-loss_{loss:.6f}-{args.model_file}"


def save_model(model, args, experiment_folder, epoch, average_loss, loss_per_epoch, *, kept=None, prune=True):
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
        obsolete = kept.pop()[2]
        if prune:
            obsolete.unlink()


def save_single_model(model, args, experiment_folder, epoch, average_loss, loss_per_epoch):
    path = _checkpoint_path(Path(experiment_folder) / "eval_models", args, epoch, average_loss)
    identical = Path(experiment_folder) / "models" / path.name
    if identical.is_file():
        publish_file(identical, path)
    else:
        save_checkpoint(path, model, args, epoch, loss_per_epoch)


def _publish(stage, destination, model_file):
    for name in ("models", "eval_models"):
        target = destination / name
        target.mkdir(exist_ok=True)
        files = list((stage / name).glob("epoch_*.pt"))
        names = {p.name for p in files}
        for path in files:
            publish_file(path, target / path.name)
        for old in target.glob("epoch_*"):
            if old.name.endswith("-" + model_file) and old.name not in names:
                old.unlink()
    for name in ("eval_metrics.csv", "partitions.npz", "labels.tsv", "vec.tsv", "embedding.pt", "embedding.json", "training.json"):
        path = stage / name
        if path.exists():
            publish_file(path, destination / name)
        elif name in ("eval_metrics.csv", "partitions.npz", "vec.tsv", "labels.tsv"):
            (destination / name).unlink(missing_ok=True)
