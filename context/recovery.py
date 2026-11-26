import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

import torch

from context.export import atomic_save, file_sha


def fit_identity(args, graph_digest, fixed_rates, selection_identity):
    locations = {"resume", "checkpoint_interval", "output_directory", "experiment_id", "graph_file", "concept_map",
                 "relationship_file_path", "hierarchy_file_path", "concept_id_file"}
    settings = {name: value for name, value in vars(args).items() if name not in locations}
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(path.name.encode())
        implementation.update(path.read_bytes())
    payload = {"settings": settings, "graph_sha256": graph_digest, "fixed_rates": fixed_rates,
               "selection": selection_identity, "source_sha256": implementation.hexdigest(),
               "torch": str(torch.__version__), "threads": args.threads or torch.get_num_threads()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def matching_stage(destination, identity):
    candidates = [stage for stage in destination.glob(".context-fit-*")
                  if (stage / "identity").is_file() and (stage / "identity").read_text() == identity
                  and (stage / "progress.pt").is_file()]
    return max(candidates, key=lambda stage: (stage / "progress.pt").stat().st_mtime_ns, default=None)


def new_stage(destination, identity):
    stage = Path(mkdtemp(prefix=".context-fit-", dir=destination))
    (stage / "identity").write_text(identity)
    return stage


def save_progress(stage, model, optimizer, order, negatives, scheduler, stopper, state):
    payload = {"identity": (stage / "identity").read_text(), "model": {
        key: getattr(value, "tensor", value).detach() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "order": order.get_state(), "negatives": negatives.bit_generator.state,
        "scheduler": scheduler.state_dict(), "stopper": vars(stopper), "state": state}
    atomic_save(payload, stage / "progress.pt")


def recover(stage, model, optimizer, order, negatives, scheduler, stopper):
    saved = torch.load(stage / "progress.pt", map_location="cpu", weights_only=True)
    if saved["identity"] != (stage / "identity").read_text():
        raise ValueError("Recovery checkpoint belongs to another run")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    vars(stopper).update(saved["stopper"])
    order.set_state(saved["order"])
    negatives.bit_generator.state = saved["negatives"]
    return saved["state"]


def publish_file(source, destination):
    temporary = destination.with_name("." + destination.name + ".pending")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def completed_records(destination, identity):
    summary = destination / "training.json"
    if not summary.is_file():
        return None
    record = json.loads(summary.read_text())
    if record.get("fit_identity") != identity:
        return None
    for name, expected in record["output_sha256"].items():
        if not (destination / name).is_file() or file_sha(destination / name) != expected:
            raise ValueError(f"Completed artifact has changed: {name}")
    return record["evaluation_records"]
