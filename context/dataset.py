import torch
import rustworkx as rx
from torch.utils.data import Dataset

class GraphEmbeddingDataset(Dataset):
    def __init__(
            self,
            graph: rx.PyDiGraph,
            device: torch.device = torch.device("cpu"),
            num_negative_samples: int = 10):
        super().__init__()

        self.edges_list = torch.as_tensor(list(graph.edge_list()), device=device)
        self.device = device
        self.num_nodes = graph.num_nodes()
        self.num_edges = graph.num_edges()
        self.num_negative_samples = num_negative_samples
        self.adjacency_sets = self.generate_adjacency_sets()
        self.non_adjacency_sets = self.generate_non_adjacency_sets()
        self.adjacency_matrix = self.create_sparse_adjacency_matrix()

    def __len__(self) -> int:
        return self.num_edges

    def create_sparse_adjacency_matrix(self):
        values = torch.ones(size=(self.edges_list.shape[0],), dtype=torch.float32, device=self.device)
        adjacency_matrix = torch.sparse_coo_tensor(
            indices=self.edges_list.T,
            values=values,
            size=(self.num_nodes, self.num_nodes),
            device=self.device
        )
        adjacency_matrix = adjacency_matrix.coalesce()
        return adjacency_matrix

    def generate_adjacency_sets(self):
        adjacency_sets = [set() for _ in range(self.num_nodes)]
        for edge in self.edges_list.cpu().numpy():
            u, v = edge
            adjacency_sets[u].add(v)
        return adjacency_sets

    def generate_non_adjacency_sets(self):
        non_adjacency_sets = [set(range(self.num_nodes)) for _ in range(self.num_nodes)]
        for node, adjacents in enumerate(self.adjacency_sets):
            non_adjacency_sets[node] -= adjacents
            non_adjacency_sets[node].discard(node)
        return non_adjacency_sets

    def sample_negative_uniform(self, batch_size: int):
        negative_nodes = torch.randint(low=0, high=self.num_nodes,
                                       size=(batch_size * self.num_negative_samples,),
                                       device=self.device)
        return negative_nodes

    def __getitem__(self, idx: int):
        batch_rel = self.edges_list[idx]
        source_nodes = batch_rel[:, 0]
        target_nodes = batch_rel[:, 1]
        negative_target_nodes =  self.sample_negative_uniform(
            batch_size=source_nodes.size(0))
        source_nodes_expanded = source_nodes.unsqueeze(1).expand(-1, self.num_negative_samples).reshape(-1)
        negative_edge_indices = torch.stack([source_nodes_expanded, negative_target_nodes], dim=0)
        negative_edge_indices_in_adj = self.adjacency_matrix._indices()
        adj_edge_keys = negative_edge_indices_in_adj[0] * self.num_nodes + negative_edge_indices_in_adj[1]
        negative_edge_keys = negative_edge_indices[0] * self.num_nodes + negative_edge_indices[1]
        mask = ~torch.isin(negative_edge_keys, adj_edge_keys)
        valid_negative_source_nodes = source_nodes_expanded[mask]
        valid_negative_target_nodes = negative_target_nodes[mask]
        positive_edges = torch.stack([source_nodes, target_nodes], dim=1)
        negative_edges = torch.stack([valid_negative_source_nodes, valid_negative_target_nodes], dim=1)
        positive_labels = torch.ones(positive_edges.size(0), device=self.device)
        negative_labels = torch.zeros(negative_edges.size(0), device=self.device)
        edges = torch.cat([positive_edges, negative_edges], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0).bool()
        return edges, labels
