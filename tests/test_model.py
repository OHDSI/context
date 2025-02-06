
from context.model import PoincareEmbedding
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from hypll.manifolds.poincare_ball import Curvature, PoincareBall

def test_model(dummy_dataset):
    batch_size = 64
    embedding_dim = 3
    curvature = 1.0

    dataloader = DataLoader(dummy_dataset, sampler=BatchSampler(sampler=RandomSampler(dummy_dataset), batch_size=batch_size,
                                                          drop_last=False))

    poincare_ball = PoincareBall(c=Curvature(curvature))

    model = PoincareEmbedding(
        num_embeddings=dummy_dataset.num_nodes,
        embedding_dim=embedding_dim,
        manifold=poincare_ball,
        root_concept_index=0
    )