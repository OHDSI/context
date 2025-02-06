
from context.model import PoincareEmbedding
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
import torch

def test_root_concept_at_origin(dummy_dataset):
    embedding_dim = 3
    curvature = 1.0
    root_concept_index = 0
    poincare_ball = PoincareBall(c=Curvature(curvature))
    model = PoincareEmbedding(
        num_embeddings=dummy_dataset.num_nodes,
        embedding_dim=embedding_dim,
        manifold=poincare_ball,
        root_concept_index=root_concept_index
    )
    root_concept = model.weight.data[model.root_concept_index]
    zero_tensor = torch.zeros(embedding_dim, device=root_concept.device, dtype=root_concept.dtype)
    assert torch.equal(root_concept, zero_tensor), "Root concept is not a zero tensor."