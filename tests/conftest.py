import pytest
import polars as pl
import torch

from context.graph import Graph
from context.dataset import GraphEmbeddingDataset

@pytest.fixture
def dummy_data_small():
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
def dummy_data_large(tree_depth: int = 10):
    def number_to_letters(n: int):
        result = ""
        while n:
            n, r = divmod(n - 1, 26)
            result = chr(65 + r) + result
        return result
    def label_gen():
        i = 1
        while True:
            yield number_to_letters(i)
            i += 1
    gen = label_gen()
    edges = []
    root = next(gen)
    def add_children(parent: str, current_depth: int):
        if current_depth < tree_depth:
            left = next(gen)
            right = next(gen)
            edges.append((parent, left, "subsumes"))
            edges.append((parent, right, "subsumes"))
            add_children(left, current_depth + 1)
            add_children(right, current_depth + 1)
    add_children(root, 1)
    return pl.DataFrame(edges, schema=["source", "target", "edge_data"], orient="row")

@pytest.fixture
def dummy_graph(dummy_data_small):
    g = Graph(dummy_data_small)
    return g.full_graph

@pytest.fixture
def dummy_graph_large(dummy_data_large):
    g = Graph(dummy_data_large)
    return g.full_graph

@pytest.fixture
def dummy_dataset(dummy_graph_large):
    num_negative_samples = 3
    dataset = GraphEmbeddingDataset(dummy_graph_large, device=torch.device('cpu'), num_negative_samples=num_negative_samples)
    return dataset