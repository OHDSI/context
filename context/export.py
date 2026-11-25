"""Portable, row-aligned embeddings and model checkpoints."""
from numbers import Integral
from pathlib import Path

import polars as pl
import torch

from context.eval import _check_points
from context.model import create_embedding


def metadata(model):
    return {"geometry": model.geometry,
            "curvature": None if model.curvature is None else model.curvature.item(),
            "embedding_dim": model.embedding_dim, "root_index": model.root_concept_index}


def save_checkpoint(path, model, args, epoch, losses):
    # hypll state_dict can contain a ManifoldTensor. Store ordinary CPU tensors so
    # both checkpoints load with weights_only=True and without a device dependency.
    state = {k: getattr(v, "tensor", v).detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "args": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
                "epoch": epoch, "losses": torch.as_tensor(losses[:epoch]).cpu(),
                **metadata(model)}, path)


def load_model(path, device="cpu"):
    """Reconstruct a model saved by this version, using its recorded metric."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weight = checkpoint["state_dict"]["weight"]
    with torch.random.fork_rng(devices=[]):
        model = create_embedding(len(weight), checkpoint["embedding_dim"], checkpoint["geometry"],
                                 checkpoint["curvature"], checkpoint["root_index"])
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


def export_embedding(folder, model, concept_ids, labels, epoch):
    values = model.coordinates().detach().cpu().contiguous()
    info = metadata(model)
    _check_points(values, info["curvature"])
    root = model.root_concept_index
    if root is not None and torch.count_nonzero(values[root]):
        raise ValueError("The anchored root moved")
    if all(isinstance(i, Integral) for i in concept_ids):
        ids = torch.tensor([int(i) for i in concept_ids], dtype=torch.long)
    elif all(isinstance(i, str) for i in concept_ids):
        ids = [str(i) for i in concept_ids]
    else:
        raise ValueError("Concept labels must be integer IDs or a list of string labels")
    torch.save({"format_version": 1, **info, "epoch": epoch, "concept_ids": ids,
                "embeddings": values}, folder / "embedding.pt")
    pl.DataFrame(values.numpy(), orient="row").write_csv(folder / "vec.tsv", separator="\t", include_header=False)
    pl.DataFrame({"label": labels}).write_csv(folder / "labels.tsv", separator="\t", include_header=False)
