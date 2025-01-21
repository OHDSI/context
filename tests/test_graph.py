from context.graph import Graph

import matplotlib.pyplot as plt
import rustworkx as rx
import polars as pl

def create_test_edge_list():
    data = [
        (1, 2, "subsumes"),
        (2, 3, "subsumes"),
        (2, 4, "subsumes"),
        (2, 5, "subsumes"),
        (3, 7, "subsumes"),
        (5, 6, "subsumes"),
        (6, 7, "subsumes"),
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
    edge_list = create_test_edge_list()
    g = Graph(edge_list)
    assert isinstance(g.full_graph, rx.PyDiGraph), "Graph is not a directed graph"
    assert len(g.full_graph.edge_list()) == 7, "Number of edges is not 7"
    assert len(g.full_graph.nodes()) == 7, "Number of nodes is not 7"

def test_create_intermediate_subgraph():
    concept_ids = [1, 3, 5, 7]
    edge_list = create_test_edge_list()
    g = Graph(edge_list)
    g.intermediate_subgraph(concept_ids)
    assert len(g.subgraph['intermediate'].nodes()) == 6, "Number of edges is not 4"
    assert len(g.subgraph['intermediate'].edge_list()) == 6, "Number of edges is not 6"
