import json
import math
from dataclasses import replace
from pathlib import Path

from context.args import Args
from context.export import file_sha
from context.train import _args_with_defaults, _train, output_folder


def refit(args, learning_rates, selection_identity):
    rates = [float(rate) for rate in learning_rates]
    if not rates or not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise ValueError("Refit requires a fixed positive learning-rate trace")
    settings = replace(_args_with_defaults(args), validation_fraction=0, epochs=len(rates),
                       eval_interval=0, retrieval_direction="observed")
    if settings.geometry == "both":
        raise ValueError("Choose the selected geometry for the full-graph refit")
    _train(settings, rates, selection_identity)
    return json.loads((output_folder(settings, settings.geometry) / "training.json").read_text())


def refit_selected(selection_folder, args):
    selection_folder = Path(selection_folder)
    selection_path = selection_folder / "training.json"
    record = json.loads(selection_path.read_text())
    if record["training_scope"] != "held_out_edge_selection":
        raise ValueError("Refit requires a validation-selected run")
    saved = Args(**record["settings"])
    selected = replace(saved, output_directory=args.output_directory, experiment_id=args.experiment_id,
                       graph_file=args.graph_file or saved.graph_file, device=args.device, resume=args.resume,
                       concept_map=args.concept_map or saved.concept_map, refit_from=None)
    if args.graph_file is None and (not selected.graph_file or not Path(selected.graph_file).is_file()):
        local_base = selection_folder
        if saved.geometry == "euclidean":
            local_base = selection_folder.parents[1 if saved.euclidean_root == "trainable" else 0]
        candidate = local_base / "graph.pkl"
        if not candidate.is_file():
            raise ValueError("Supply graph_file to locate the graph used for selection")
        selected.graph_file = str(candidate)
    if output_folder(selected, saved.geometry).resolve() == selection_folder.resolve():
        raise ValueError("Use a separate experiment for the full-graph refit")
    epoch = record["selected_epoch"]
    return refit(selected, record["learning_rates"][:epoch], {"selection_sha256": file_sha(selection_path),
                 "graph_sha256": record["graph_sha256"], "geometry": saved.geometry, "selected_epoch": epoch})
