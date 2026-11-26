import hashlib
import json
from numbers import Integral
from pathlib import Path

import polars as pl
import torch

from context.eval import _check_points
from context.model import create_embedding


def metadata(model):
    return {"geometry": model.geometry,
            "curvature": None if model.curvature is None else model.curvature.item(),
            "distance_scale": getattr(model, "distance_scale", 1.0),
            "embedding_dim": model.embedding_dim, "root_index": model.root_concept_index}


def save_checkpoint(path, model, args, epoch, losses):
    state = {k: getattr(v, "tensor", v).detach().cpu() for k, v in model.state_dict().items()}
    atomic_save({"state_dict": state, "args": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
                "epoch": epoch, "losses": torch.as_tensor(losses[:epoch]).cpu(),
                **metadata(model)}, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weight = checkpoint["state_dict"]["weight"]
    with torch.random.fork_rng(devices=[]):
        model = create_embedding(len(weight), checkpoint["embedding_dim"], checkpoint["geometry"],
                                 checkpoint["curvature"], checkpoint["root_index"], checkpoint.get("distance_scale", 1.0))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def concept_labels(concept_ids, concept_map):
    if concept_map is None:
        return concept_ids
    wanted = [str(i) for i in concept_ids]
    names = (pl.scan_csv(concept_map, separator="\t", quote_char=None, infer_schema_length=0)
             .select("concept_id", "concept_name").filter(pl.col("concept_id").is_in(wanted)).collect())
    lookup = dict(names.iter_rows())
    return [lookup.get(str(i)) or str(i) for i in concept_ids]


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_export(ids, values, info):
    if info["geometry"] not in ("euclidean", "hyperbolic") or values.ndim != 2 or not values.is_floating_point():
        raise ValueError("Invalid embedding geometry or coordinates")
    if isinstance(ids, torch.Tensor):
        if ids.ndim != 1 or ids.dtype != torch.int64 or len(ids.unique()) != len(ids):
            raise ValueError("Expected unique integer concept IDs")
    elif not isinstance(ids, list) or not all(isinstance(label, str) for label in ids) or len(set(ids)) != len(ids):
        raise ValueError("Expected unique string labels")
    if tuple(values.shape) != (len(ids), info["dimensions"]) or info["nodes"] != len(ids):
        raise ValueError("Embedding dimensions and concept IDs disagree")
    curvature = info["effective_curvature_magnitude"]
    if (curvature is None) != (info["geometry"] == "euclidean"):
        raise ValueError("Curvature and embedding geometry disagree")
    for key in ("root_index", "graph_root_index"):
        row = info.get(key)
        if row is not None and (not isinstance(row, int) or not 0 <= row < len(ids)):
            raise ValueError("Root row outside the embedding table")
    if info.get("root_index") is not None and torch.count_nonzero(values[info["root_index"]]):
        raise ValueError("The anchored root moved")
    for block in values.split(32768):
        _check_points(block.double(), curvature)


def load_embedding(path, *, geometry=None, curvature=None):
    path = Path(path)
    sidecar = path.with_suffix(".json")
    expected = json.loads(sidecar.read_text()) if sidecar.exists() else None
    with path.open("rb") as stream:
        if expected is not None:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected.pop("payload_sha256"):
                raise ValueError("Embedding checksum mismatch")
            stream.seek(0)
        result = torch.load(stream, weights_only=True, map_location="cpu")
    info = result.get("metadata")
    if info is None and expected is None and "format_version" not in result:
        if geometry not in ("euclidean", "hyperbolic") or (geometry == "hyperbolic") != (curvature is not None):
            raise ValueError("Declare geometry and effective Poincaré curvature for an unlabelled legacy embedding")
        ids, points = result["concept_ids"], result["embeddings"]
        if isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[1] == 1:
            result["concept_ids"] = ids[:, 0]
        info = {"geometry": geometry, "effective_curvature_magnitude": curvature,
                "nodes": len(ids), "dimensions": points.shape[1], "provenance": "caller-supplied geometry"}
        result["metadata"] = info
    elif info is None:
        if expected is not None or result.get("format_version") != 1:
            raise ValueError("Unknown embedding format")
        info = {"geometry": result["geometry"], "effective_curvature_magnitude": result["curvature"],
                "root_index": result.get("root_index"), "graph_root_index": result.get("graph_root_index"),
                "distance_scale": result.get("distance_scale", 1.0),
                "dimensions": result["embedding_dim"], "nodes": len(result["concept_ids"]),
                "selected_epoch": result["epoch"]}
        result["metadata"] = info
    elif info != expected:
        raise ValueError("Missing or mismatched embedding metadata sidecar")
    if geometry is not None and info["geometry"] != geometry:
        raise ValueError("Embedding geometry differs from the requested geometry")
    if curvature is not None and info["effective_curvature_magnitude"] != curvature:
        raise ValueError("Embedding curvature differs from the requested curvature")
    _validate_export(result["concept_ids"], result["embeddings"], info)
    return result


def export_embedding(folder, model, concept_ids, labels, epoch, *, graph_root_index=None, provenance=None, write_tsv=True):
    values = model.coordinates().detach().cpu().contiguous()
    info = metadata(model)
    root = model.root_concept_index
    if root is not None and torch.count_nonzero(values[root]):
        raise ValueError("The anchored root moved")
    if all(isinstance(i, Integral) for i in concept_ids):
        ids = torch.tensor(concept_ids, dtype=torch.long)
    elif all(isinstance(i, str) for i in concept_ids):
        ids = list(concept_ids)
    else:
        raise ValueError("Concept labels must be integer IDs or a list of string labels")
    if len(labels) != len(ids):
        raise ValueError("Expected a display label for each concept")
    graph_root_index = root if graph_root_index is None else graph_root_index
    manifest = {**(provenance or {}), "schema": "ohdsi-context-embedding/1", "geometry": model.geometry,
                "coordinate_model": "poincare_ball" if model.geometry == "hyperbolic" else "euclidean",
                "effective_curvature_magnitude": info["curvature"], "distance_scale": info["distance_scale"],
                "dimensions": info["embedding_dim"], "nodes": len(ids), "dtype": str(values.dtype),
                "selected_epoch": epoch, "graph_root_index": graph_root_index, "root_index": root}
    _validate_export(ids, values, manifest)
    output = folder / "embedding.pt"
    atomic_save({"format_version": 1, **info, "graph_root_index": graph_root_index, "epoch": epoch,
                 "concept_ids": ids, "embeddings": values, "metadata": manifest}, output)
    checksum = file_sha(output)
    output.with_suffix(".json").write_text(json.dumps({**manifest, "payload_sha256": checksum}, indent=2) + "\n")
    if write_tsv:
        pl.DataFrame(values.numpy(), orient="row").write_csv(folder / "vec.tsv", separator="\t", include_header=False)
        pl.DataFrame({"label": labels}).write_csv(folder / "labels.tsv", separator="\t", include_header=False)
    return checksum
