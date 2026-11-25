from functools import cached_property

import numpy as np
import rustworkx as rx
import torch
from torch.utils.data import Dataset, get_worker_info


class GraphEmbeddingDataset(Dataset):
    def __init__(self, graph: rx.PyDiGraph, device=torch.device("cpu"), num_negative_samples=10,
                 directed=False, seed=None):
        self.device = device
        self.directed = directed
        self.num_negative_samples = num_negative_samples
        if num_negative_samples < 1:
            raise ValueError("At least one negative sample is required")
        indices = list(graph.node_indices())
        self.concept_ids = [graph[i] for i in indices]
        self.num_nodes = len(indices)
        if len(set(self.concept_ids)) != self.num_nodes:
            raise ValueError("Graph node labels must be unique")
        position = {index: row for row, index in enumerate(indices)}
        pairs = [(position[s], position[t]) for s, t in graph.edge_list() if s != t]
        if not pairs:
            raise ValueError("The graph has no non-self edges")
        pairs = np.asarray(pairs, dtype=np.int64)
        _, first = np.unique(pairs, axis=0, return_index=True)
        edges = pairs[np.sort(first)]
        self._observed = edges
        self.edges_list = torch.as_tensor(edges, dtype=torch.long, device=device)
        self.num_edges = len(edges)
        positives = edges if directed else np.unique(np.concatenate((edges, edges[:, ::-1])), axis=0)

        # For a source with sorted blocked targets b[j], the r-th allowed node
        # is r + count(b[j] - j <= r). Index the shifted lists once for all rows.
        nodes = np.arange(self.num_nodes, dtype=np.int64)
        blocked = np.sort(np.r_[positives[:, 0] * self.num_nodes + positives[:, 1],
                                nodes * self.num_nodes + nodes])
        counts = np.bincount(blocked // self.num_nodes, minlength=self.num_nodes)
        self._offsets = np.cumsum(counts) - counts
        self._available = self.num_nodes - counts
        self._shifted = blocked - (np.arange(len(blocked)) - np.repeat(self._offsets, counts))
        self._positives = positives[self._available[positives[:, 0]] > 0]
        self.skipped_positives = len(positives) - len(self._positives)
        self._rng = np.random.default_rng(seed)
        self._worker_id = None

    def __len__(self):
        return self.num_edges

    @property
    def training_size(self):
        return len(self._positives)

    def create_sparse_adjacency_matrix(self, directed):
        indices = self.edges_list.T
        if not directed:
            indices = torch.unique(torch.cat((indices, indices.flip(0)), dim=1), dim=1)
        return torch.sparse_coo_tensor(indices, torch.ones(indices.size(1), device=self.device),
                                       (self.num_nodes, self.num_nodes)).coalesce()

    @cached_property
    def directed_adjacency_matrix(self):
        return self.create_sparse_adjacency_matrix(True)

    @cached_property
    def undirected_adjacency_matrix(self):
        return self.create_sparse_adjacency_matrix(False)

    def generate_adjacency_sets(self, directed):
        adjacency = [set() for _ in range(self.num_nodes)]
        for source, target in self.edges_list.cpu().tolist():
            adjacency[source].add(target)
            if not directed:
                adjacency[target].add(source)
        return adjacency

    def sample_negative_uniform(self, batch_size):
        """Unconditional draws, retained for callers of the original helper."""
        return torch.randint(self.num_nodes, (batch_size * self.num_negative_samples,), device=self.device)

    def _sample(self, positives, rng):
        sources = positives[:, :1]
        if rng is None:
            worker = get_worker_info()
            if worker is not None and self._worker_id != worker.id:
                self._rng = np.random.default_rng(worker.seed)
                self._worker_id = worker.id
            rng = self._rng
        ranks = rng.integers(self._available[sources], size=(len(sources), self.num_negative_samples))
        ends = np.searchsorted(self._shifted, sources * self.num_nodes + ranks, side="right")
        return ranks + ends - self._offsets[sources]

    def candidate_nodes(self, indices, rng=None):
        """Draw K conditional negatives per positive indexed in [0, training_size)."""
        positives = self._positives[indices].reshape(-1, 2)
        negatives = self._sample(positives, rng)
        return (torch.as_tensor(positives[:, :1], dtype=torch.long, device=self.device),
                torch.as_tensor(np.column_stack((positives[:, 1], negatives)), device=self.device))

    def __getitem__(self, idx):
        # Preserve original edge indexing and the flat positives-first layout.
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        observed = self._observed[idx].reshape(-1, 2)
        eligible = observed[self._available[observed[:, 0]] > 0]
        negatives = self._sample(eligible, None)
        pairs = np.stack((np.broadcast_to(eligible[:, :1], negatives.shape), negatives), axis=-1)
        edges = torch.as_tensor(np.concatenate((observed, pairs.reshape(-1, 2))), device=self.device)
        labels = torch.arange(len(edges), device=self.device) < len(observed)
        return edges, labels
