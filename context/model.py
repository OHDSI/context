from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
import hypll.nn as hnn
import torch


class PoincareEmbedding(hnn.HEmbedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            manifold: PoincareBall,
            root_concept_index: int
    ):
        super().__init__(num_embeddings, embedding_dim, manifold)
        self._init_uniform()
        self.root_concept_index = root_concept_index

        if root_concept_index is not None:
            with torch.no_grad():
                self.weight.data[self.root_concept_index].zero_()

            self.weight.tensor.register_hook(self._zero_root_grad)

    def _zero_root_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad[self.root_concept_index].zero_()
        return grad

    def _init_uniform(self):
        with torch.no_grad():
            self.weight.tensor.uniform_(-0.001, 0.001)
            new_tangent = TangentTensor(
                data=self.weight.tensor,
                manifold_points=None,
                manifold=self.manifold,
                man_dim=-1,
            )
            self.weight.copy_(self.manifold.expmap(new_tangent).tensor)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(edges)
        edge_distances = self.manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances
