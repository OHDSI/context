from context.dataset import GraphEmbeddingDataset
import torch
import pytest

def test_adjacency_matrix(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph)
    adj_matrix = dataset.adjacency_matrix
    edges_list = dataset.edges_list.cpu().numpy()
    dense_adj_matrix = adj_matrix.to_dense()
    for edge in edges_list:
        u, v = edge
        assert dense_adj_matrix[u, v] == 1, f"Edge ({u}, {v}) not present in adjacency matrix"

def test_generate_adjacency_sets(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph)
    adjacency_sets = dataset.generate_adjacency_sets()
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

def test_generate_non_adjacency_sets(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph)
    non_adjacency_sets = dataset.generate_non_adjacency_sets()
    expected_non_adjacency_sets = [
        {2, 3, 4, 5, 6},    # A (0) -> B (1)
        {0, 5, 6},          # B (1) -> C (2), D (3), E (4)
        {0, 1, 3, 4, 6},    # C (2) -> G (5)
        {0, 1, 2, 4, 5, 6}, # D (3) has no outgoing edges
        {0, 1, 2, 3, 5},    # E (4) -> F (6)
        {0, 1, 2, 3, 4, 6}, # G (5) has no outgoing edges
        {0, 1, 2, 3, 4}     # F (6) -> G (5)
    ]
    for i, actual_set in enumerate(non_adjacency_sets):
        assert actual_set == expected_non_adjacency_sets[i], f"Node {i} has the wrong non-adjacency set: {actual_set}"

def test_negative_samples_insufficient(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph, device=torch.device('cpu'), num_negative_samples=10)
    with pytest.raises(ValueError, match="Not enough non-connected nodes to sample"):
        for idx in range(len(dataset)):
            edge, labels = dataset[idx]

def test_get_item(dummy_graph):
    dataset = GraphEmbeddingDataset(dummy_graph, device=torch.device('cpu'), num_negative_samples=3)
    idx = 2
    edges, labels = dataset[idx]
    assert len(labels) == 4, "Not four elements"
    assert labels[0] == True, "First element not True"
    assert labels[1:].all() == False, "Second to last elements not False"