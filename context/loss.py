import torch
from torch.nn import functional as F


def candidate_loss(distances: torch.Tensor) -> torch.Tensor:
    if distances.ndim != 2 or not len(distances) or distances.shape[1] < 2:
        raise ValueError("Expected [positive edges, 1 + negative samples] distances")
    return torch.logsumexp(distances[:, :1] - distances, dim=-1).mean()


def poincare_embeddings_loss(distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.shape != distances.shape:
        raise ValueError("Targets and distances must have matching shapes")
    targets = targets.bool()
    if not distances.numel() or not targets.any(dim=-1).all():
        raise ValueError("Every candidate group requires a positive")
    logits = -distances
    offset = logits.max(dim=-1, keepdim=True).values
    logits = logits - offset
    positive = logits.masked_fill(~targets, -torch.inf)
    return (torch.logsumexp(logits, dim=-1) - torch.logsumexp(positive, dim=-1)).mean()


def poincare_original_loss(distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.bool()
    if not targets.any() or targets.all():
        raise ValueError("Positive and negative targets are required")
    return (F.softplus(distances[targets]).mean() + F.softplus(-distances[~targets]).mean()) / 2
