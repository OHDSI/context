from context.dataset import GraphEmbeddingDataset
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import torch

def test_directed_adjacency_matrix(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph)
    adj_matrix = dataset.directed_adjacency_matrix
    edges_list = dataset.edges_list.cpu().numpy()
    dense_adj_matrix = adj_matrix.to_dense()
    for edge in edges_list:
        u, v = edge
        assert dense_adj_matrix[u, v] == 1, f"Edge ({u}, {v}) not present in adjacency matrix"

def test_undirected_adjacency_matrix(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph)
    adj_matrix = dataset.undirected_adjacency_matrix
    edges_list = dataset.edges_list.cpu().numpy()
    dense_adj_matrix = adj_matrix.to_dense()
    for edge in edges_list:
        u, v = edge
        assert dense_adj_matrix[u, v] == 1, f"Edge ({u}, {v}) not present in adjacency matrix"
        assert dense_adj_matrix[v, u] == 1, f"Edge ({v}, {u}) not present in adjacency matrix"

def test_directed_adjacency_sets(dummy_graph):
    directed = True
    dataset = GraphEmbeddingDataset(dummy_graph)
    adjacency_sets = dataset.generate_adjacency_sets(directed)
    expected_adjacency_sets = [
        {1},        # A(0) -> B(1)
        {2, 3, 4},  # B(1) -> C(2), D(3), E(4)
        {5},        # C(2) -> G(5)
        set(),      # D(3) has no outgoing edges
        {6},        # E(4) -> F(6)
        set(),      # G(5) has no outgoing edges
        {5}         # F(6) -> G(5)
    ]
    for i, actual_set in enumerate(adjacency_sets):
        assert actual_set == expected_adjacency_sets[i], f"Node {i} has the wrong adjacency set: {actual_set}"

def test_undirected_adjacency_sets(dummy_graph):
    directed = False
    dataset = GraphEmbeddingDataset(dummy_graph)
    adjacency_sets = dataset.generate_adjacency_sets(directed)
    expected_adjacency_sets = [
        {1},            # A(0) -> B(1)
        {0, 2, 3, 4},   # B(1) -> A(0), C(2), D(3), E(4)
        {1, 5},         # C(2) -> B(1), G(5)
        {1},            # D(3) -> B(1)
        {1, 6},         # E(4) -> F(6), B(1)
        {2, 6},         # G(5) -> C(2), F(6)
        {4, 5}          # F(6) -> G(5), E(4)
    ]
    for i, actual_set in enumerate(adjacency_sets):
        assert actual_set == expected_adjacency_sets[i], f"Node {i} has the wrong adjacency set: {actual_set}"

def test_get_item(dummy_graph):
    num_negative_samples = 3
    dataset = GraphEmbeddingDataset(dummy_graph, device=torch.device('cpu'), num_negative_samples=num_negative_samples, directed=False)
    idx = [0, 1, 2]
    edges, labels = dataset[idx]
    # print(f"Edges: {edges}")
    # print(f"Labels: {labels}")
    assert labels.shape[0] >= len(idx), "Not enough edges returned"
    assert torch.all(labels[:len(idx)] == True).item(), f"First {len(idx)} labels must be True"
    if labels.shape[0] > len(idx):
            assert torch.all(labels[len(idx):] == False).item(), "Remaining labels must be False"

def test_dataloader(dummy_graph_large):
    batch_size = 8
    num_negative_samples = 10
    dataset = GraphEmbeddingDataset(dummy_graph_large, num_negative_samples=num_negative_samples)
    dataloader = DataLoader(dataset, sampler=BatchSampler(sampler=RandomSampler(dataset), batch_size=batch_size,
                                                          drop_last=False))
    for edges, labels in dataloader:
        positive_label_count = labels.bool().sum().item()
        # print(f"Edges: {edges}")
        # print(f"Labels: {labels}")
        assert positive_label_count == batch_size, (
            f"Expected {batch_size} positive labels, got {positive_label_count}"
        )
        break
