import torch


def poincare_embeddings_loss(distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = distances.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss

def poincare_original_loss(distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive_loss = -torch.log(torch.sigmoid(-distances[targets]))
    negative_loss = -torch.log(torch.sigmoid(distances[~targets]))
    loss = (positive_loss.mean() + negative_loss.mean()) / 2
    return loss