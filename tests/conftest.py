import pytest
import polars as pl
from context.graph import Graph

@pytest.fixture
def dummy_data():
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

@pytest.fixture
def dummy_graph(dummy_data):
    g = Graph(dummy_data)
    g.plot()
    return g.full_graph
