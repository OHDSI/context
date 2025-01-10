from context.graph import Graph
import networkx as nx


def test_addition():
    assert (1 + 1) == 2


def test_directed_graph():
    concept_ids = [1, 2, 3, 4]
    hierarchy = [(1, 2), (2, 3), (3, 4)]
    g = Graph(concept_ids, hierarchy)
    assert isinstance(g.full_graph, nx.DiGraph), "Graph is not a directed graph"
