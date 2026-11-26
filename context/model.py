import math

import torch
from hypll import nn as hnn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from torch.nn import functional as F


def _ball_distance(chord, left_margin, right_margin, curvature):
    sqrt_c = curvature ** 0.5
    return (2 / sqrt_c) * torch.asinh(sqrt_c * chord / (left_margin.sqrt() * right_margin.sqrt()))


def _margin(points, curvature):
    return 1 - curvature * points.square().sum(dim=-1)


def pairwise_distance(left, right, curvature=None):
    chord = torch.cdist(left, right, compute_mode="donot_use_mm_for_euclid_dist")
    if curvature is None:
        return chord
    return _ball_distance(chord, _margin(left, curvature)[:, None],
                          _margin(right, curvature)[None, :], curvature)


class _EdgeDistances:
    def forward(self, edges: torch.Tensor, targets=None) -> torch.Tensor:
        if targets is None:
            sources, targets = edges.unbind(dim=-1)
        else:
            sources = edges
        left = self._lookup(sources)
        right = self._lookup(targets)
        chord = torch.linalg.vector_norm(left - right, dim=-1)
        c = self.curvature
        if c is None:
            return chord if self.distance_scale == 1.0 else chord * self.distance_scale
        return _ball_distance(chord, _margin(left, c), _margin(right, c), c)

    def _lookup(self, indices):
        points = F.embedding(indices, self.coordinates(), padding_idx=self.root_concept_index,
                             sparse=getattr(self, "sparse_gradients", False))
        if points.device.type == "mps" and self.root_concept_index is not None:
            points = points.masked_fill((indices == self.root_concept_index).unsqueeze(-1), 0.0)
        return points

    def _anchor(self, root_concept_index):
        if root_concept_index is not None:
            if not 0 <= root_concept_index < self.num_embeddings:
                raise ValueError("Root index is outside the embedding table")
            with torch.no_grad():
                self.coordinates()[root_concept_index].zero_()
        self.root_concept_index = root_concept_index


class PoincareEmbedding(_EdgeDistances, hnn.HEmbedding):
    geometry = "hyperbolic"

    def __init__(self, num_embeddings, embedding_dim, manifold: PoincareBall, root_concept_index=None):
        super().__init__(num_embeddings, embedding_dim, manifold)
        self._anchor(root_concept_index)

    def reset_parameters(self):
        with torch.no_grad():
            tangent = TangentTensor(data=torch.empty_like(self.weight.tensor).uniform_(-0.001, 0.001),
                                    manifold_points=None, manifold=self.manifold, man_dim=-1)
            self.weight.copy_(self.manifold.expmap(tangent).tensor)
            self._anchor(getattr(self, "root_concept_index", None))

    @property
    def curvature(self):
        return self.manifold.c()

    def coordinates(self):
        return self.weight.tensor


class EuclideanEmbedding(_EdgeDistances, torch.nn.Embedding):
    geometry = "euclidean"
    curvature = None

    def __init__(self, num_embeddings, embedding_dim, root_concept_index=None, distance_scale=1.0):
        if not math.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError("Euclidean distance scale must be finite and positive")
        self.distance_scale = float(distance_scale)
        super().__init__(num_embeddings, embedding_dim)
        self._anchor(root_concept_index)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.uniform_(-0.001, 0.001)
            self._anchor(getattr(self, "root_concept_index", None))

    def coordinates(self):
        return self.weight


def create_embedding(num_embeddings, embedding_dim, geometry="hyperbolic", curvature=1.0,
                     root_concept_index=None, euclidean_distance_scale=1.0):
    if num_embeddings < 1 or embedding_dim < 1:
        raise ValueError("Embedding table dimensions must be positive")
    if geometry == "euclidean":
        return EuclideanEmbedding(num_embeddings, embedding_dim, root_concept_index, euclidean_distance_scale)
    if geometry != "hyperbolic":
        raise ValueError(f"Unknown embedding geometry {geometry!r}")
    curvature = torch.tensor(float(curvature), dtype=torch.float32).item()
    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError("Poincaré curvature magnitude must be finite and positive")
    manifold = PoincareBall(c=Curvature(curvature, constraining_strategy=torch.nn.Identity()))
    return PoincareEmbedding(num_embeddings, embedding_dim, manifold, root_concept_index)
