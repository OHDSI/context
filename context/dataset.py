from copy import copy
from functools import cached_property

import numpy as np
import rustworkx as rx
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset, get_worker_info


class GraphEdges(Dataset):
    def __init__(self, graph: rx.PyDiGraph, device=torch.device("cpu"), directed=False):
        self.device = device
        self.directed = directed
        indices = np.asarray(graph.node_indices(), dtype=np.int64)
        pairs = np.asarray(graph.edge_list(), dtype=np.int64).reshape(-1, 2)
        if len(indices) and indices[-1] != len(indices) - 1:
            pairs = np.searchsorted(indices, pairs)
        self._set_edges(list(graph.nodes()), pairs)

    def with_training_edges(self, edges):
        dataset = copy(self)
        for attribute in ("directed_adjacency_matrix", "undirected_adjacency_matrix"):
            vars(dataset).pop(attribute, None)
        dataset._set_edges(self.concept_ids, edges)
        return dataset

    def _set_edges(self, concept_ids, pairs):
        self.concept_ids = concept_ids
        self.num_nodes = len(concept_ids)
        if len(set(concept_ids)) != self.num_nodes:
            raise ValueError("Graph node labels must be unique")
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs.size and (pairs.min() < 0 or pairs.max() >= self.num_nodes)):
            raise ValueError("Expected edges between existing concept rows")
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        edge_keys = pairs[:, 0] * self.num_nodes + pairs[:, 1]
        unique_keys, first = np.unique(edge_keys, return_index=True)
        self._observed = pairs[np.sort(first)] if len(first) < len(pairs) else pairs
        self.edges_list = torch.as_tensor(self._observed, dtype=torch.long, device=self.device)
        self.num_edges = len(self._observed)
        if not self.directed:
            reverse_keys = self._observed[:, 1] * self.num_nodes + self._observed[:, 0]
            unique_keys = np.union1d(unique_keys, reverse_keys)
        self.neighbor_rows = self._neighbor_index(unique_keys)
        return unique_keys

    def _neighbor_index(self, sorted_keys):
        source_rows, target_rows = divmod(sorted_keys, self.num_nodes)
        offsets = np.r_[0, np.cumsum(np.bincount(source_rows, minlength=self.num_nodes))]
        return csr_matrix((np.ones(len(target_rows), dtype=bool), target_rows, offsets),
                          shape=(self.num_nodes, self.num_nodes))

    def adjacency(self, directed):
        if directed == self.directed:
            return self.neighbor_rows
        sources, targets = self._observed.T
        keys = sources * self.num_nodes + targets
        if not directed:
            keys = np.concatenate((keys, targets * self.num_nodes + sources))
        return self._neighbor_index(np.unique(keys))

    def __len__(self):
        return self.num_edges

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


class GraphEmbeddingDataset(GraphEdges):
    def __init__(self, graph: rx.PyDiGraph, device=torch.device("cpu"), num_negative_samples=10,
                 directed=False, seed=None):
        if num_negative_samples < 1:
            raise ValueError("At least one negative sample is required")
        self.num_negative_samples = num_negative_samples
        self._seed = seed
        super().__init__(graph, device=device, directed=directed)
        self._seed = None

    def _set_edges(self, concept_ids, pairs):
        unique_keys = super()._set_edges(concept_ids, pairs)
        if not self.num_edges:
            raise ValueError("The graph has no non-self edges")
        positives = self._observed if self.directed else np.column_stack(divmod(unique_keys, self.num_nodes))
        node_rows = np.arange(self.num_nodes, dtype=np.int64)
        blocked_keys = np.union1d(unique_keys, node_rows * (self.num_nodes + 1))
        blocked_counts = np.diff(self.neighbor_rows.indptr).astype(np.int64) + 1
        self._blocked_offsets = blocked_counts.cumsum() - blocked_counts
        self._available_counts = self.num_nodes - blocked_counts
        within_row = np.arange(len(blocked_keys)) - np.repeat(self._blocked_offsets, blocked_counts)
        self._complement_thresholds = blocked_keys % self.num_nodes - within_row
        self._global_thresholds = blocked_keys - within_row
        self._positives = positives[self._available_counts[positives[:, 0]] > 0]
        self.skipped_positives = len(positives) - len(self._positives)
        self._rng = np.random.default_rng(self._seed)
        self._worker_id = None

    @property
    def training_size(self):
        return len(self._positives)

    def sample_negative_uniform(self, batch_size):
        return torch.randint(self.num_nodes, (batch_size * self.num_negative_samples,), device=self.device)

    def _sample(self, positives, rng):
        sources = positives[:, :1]
        if rng is None:
            worker = get_worker_info()
            if worker is not None and self._worker_id != worker.id:
                self._rng = np.random.default_rng(worker.seed)
                self._worker_id = worker.id
            rng = self._rng
        ranks = rng.integers(self._available_counts[sources], size=(len(sources), self.num_negative_samples))
        negatives = ranks.copy()
        counts = self.num_nodes - self._available_counts[sources[:, 0]]
        small = counts <= 16
        offsets = self._blocked_offsets[sources[:, 0]]
        if small.any():
            small_ranks = ranks[small]
            small_offsets = offsets[small]
            small_counts = counts[small]
            shift = np.zeros_like(small_ranks)
            for position in range(int(small_counts.max())):
                active = small_counts > position
                thresholds = self._complement_thresholds[small_offsets[active] + position]
                shift[active] += small_ranks[active] >= thresholds[:, None]
            negatives[small] += shift
        if not small.all():
            keys = sources[~small] * self.num_nodes + ranks[~small]
            negatives[~small] += np.searchsorted(self._global_thresholds, keys, side="right") - offsets[~small, None]
        return negatives

    def candidate_nodes(self, indices, rng=None):
        """Return source rows and target rows with the observed neighbor in column zero."""
        positives = self._positives[indices].reshape(-1, 2)
        negatives = self._sample(positives, rng)
        return (torch.as_tensor(positives[:, :1], dtype=torch.long, device=self.device),
                torch.as_tensor(np.column_stack((positives[:, 1], negatives)), device=self.device))

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        observed = self._observed[idx].reshape(-1, 2)
        eligible = observed[self._available_counts[observed[:, 0]] > 0]
        negatives = self._sample(eligible, None)
        pairs = np.stack((np.broadcast_to(eligible[:, :1], negatives.shape), negatives), axis=-1)
        edges = torch.as_tensor(np.concatenate((observed, pairs.reshape(-1, 2))), device=self.device)
        labels = torch.arange(len(edges), device=self.device) < len(observed)
        return edges, labels
