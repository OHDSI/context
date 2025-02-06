from context.graph import Graph
import rustworkx as rx

def test_create_full_graph(dummy_data_small):
    g = Graph(dummy_data_small)
    g.plot()
    assert isinstance(g.full_graph, rx.PyDiGraph), "Graph is not a directed graph"
    assert g.full_graph.num_nodes() == 7, "Number of nodes is not 7"
    assert g.full_graph.num_edges() == 7, "Number of edges is not 7"

def test_create_intermediate_subgraph(dummy_data_small):
    concept_ids = ["A", "C", "E", "G"]
    g = Graph(dummy_data_small)
    g.intermediate_subgraph(concept_ids)
    g.plot("intermediate")
    assert g.subgraph['intermediate'].num_nodes() == 6, "Number of nodes is not 6"
    assert g.subgraph['intermediate'].num_edges() == 6, "Number of edges is not 6"

def test_find_root_nodes(dummy_data_small):
    g = Graph(dummy_data_small)
    root_node = g.find_root_node()
    assert root_node is not None, "Root node is not found"
    assert root_node == "A", "Root node does not have the label A"
