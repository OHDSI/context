from context.graph import Graph
import rustworkx as rx
import polars as pl

def get_dummy_data():
    data = [
        ("A", "B", "subsumes"),
        ("B", "C", "subsumes"),
        ("B", "D", "subsumes"),
        ("B", "E", "subsumes"),
        ("C", "G", "subsumes"),
        ("E", "F", "subsumes"),
        ("F", "G", "subsumes"),
    ]
    schema = [
        "source",
        "target",
        "edge_data"
    ]
    edge_list = pl.DataFrame(
        data,
        schema,
        orient="row"
    )
    return edge_list

def test_create_full_graph():
    edge_list = get_dummy_data()
    g = Graph(edge_list)
    assert isinstance(g.full_graph, rx.PyDiGraph), "Graph is not a directed graph"
    assert g.full_graph.num_nodes() == 7, "Number of nodes is not 7"
    assert g.full_graph.num_edges() == 7, "Number of edges is not 7"

def test_create_intermediate_subgraph():
    concept_ids = ["A", "C", "E", "G"]
    edge_list = get_dummy_data()
    g = Graph(edge_list)
    g.intermediate_subgraph(concept_ids)
    assert g.subgraph['intermediate'].num_nodes() == 6, "Number of nodes is not 6"
    assert g.subgraph['intermediate'].num_edges() == 6, "Number of edges is not 6"
