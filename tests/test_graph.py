from context.graph import Graph
import networkx as nx
import polars as pl

def test_directed_graph():
    data = [
        (1, 2, 1, 1),
        (2, 3, 1, 1),
        (1, 3, 2, 2),
        (2, 4, 1, 1),
        (2, 5, 1, 1),
    ]
    schema = [
        "ancestor_concept_id",
        "descendant_concept_id",
        "min_levels_of_separation",
        "max_levels_of_separation"
    ]
    hierarchy = pl.DataFrame(
        data,
        schema,
        orient="row"
    )
    g = Graph(hierarchy)
    assert isinstance(g.full_graph, nx.DiGraph), "Graph is not a directed graph"
    assert g.full_graph.number_of_edges() == 4, "Number of edges is not 4"
    assert g.full_graph.number_of_nodes() == 5, "Number of nodes is not 5"
