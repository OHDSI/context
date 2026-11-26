import numpy as np
import pytest
import rustworkx as rx
import torch

from context.dataset import GraphEmbeddingDataset


@pytest.mark.parametrize("directed", [False, True])
def test_exact_complement_for_all_sources_with_index_gaps(directed):
    g = rx.PyDiGraph()
    g.add_nodes_from([91, 92, 93, 94, 95, 96])
    g.remove_node(1)
    g.add_edges_from_no_data([(0, 2), (0, 2), (0, 3), (3, 4), (3, 5), (4, 4)])
    ds = GraphEmbeddingDataset(g, num_negative_samples=100, directed=directed)
    assert ds.concept_ids == [91, 93, 94, 95, 96]
    assert ds.num_edges == 4
    adjacency = ds.generate_adjacency_sets(directed)
    indices = list(range(ds.training_size))
    s, targets = ds.candidate_nodes(indices, np.random.default_rng(42))
    torch.testing.assert_close(targets, ds.candidate_nodes(indices, np.random.default_rng(42))[1])
    for (source,), row in zip(s.tolist(), targets.tolist()):
        assert row[0] in adjacency[source]
        assert set(row[1:]) == set(range(ds.num_nodes)) - adjacency[source] - {source}
    edges, labels = ds[0]
    assert edges.shape == (101, 2)
    assert labels.tolist() == [True] + [False] * 100


def test_star_preserves_graph_and_skips_only_queries_without_a_negative():
    graph = rx.PyDiGraph()
    graph.add_nodes_from([10, 11, 12, 13])
    graph.add_edges_from_no_data([(0, 1), (0, 2), (0, 3)])
    dataset = GraphEmbeddingDataset(graph, directed=False)
    assert dataset.training_size == dataset.skipped_positives == 3
    edges, labels = dataset[:]
    assert labels.all() and torch.equal(edges, dataset.edges_list)
    sources, targets = dataset.candidate_nodes([0, 1, 2])
    assert set(sources.flatten().tolist()) == {1, 2, 3}
    assert targets[:, 0].tolist() == [0, 0, 0]
    assert dataset.generate_adjacency_sets(True)[0] == {1, 2, 3}
    assert GraphEmbeddingDataset(graph, directed=True).training_size == 0
