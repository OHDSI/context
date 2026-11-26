import os
from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from context.args import Args
from context.classical import METHODS
from context.export import load_embedding
from context.train import build_graph, output_folder, train


@pytest.fixture(scope="module")
def supplied_taxonomy(tmp_path_factory):
    location = os.environ.get("CONTEXT_SNOMED_VOCABULARY")
    if not location:
        pytest.skip("Set CONTEXT_SNOMED_VOCABULARY to a licensed OMOP vocabulary directory")
    folder = Path(location)
    concepts = str(folder / "CONCEPT.csv")
    relationships = str(folder / "CONCEPT_RELATIONSHIP.csv")
    graph = build_graph(Args(concept_map=concepts, relationship_file_path=relationships,
                             taxonomy_scope="all", snomed_concepts="active"))
    csv_reader = "read_csv(?, delim='\\t', header=true, all_varchar=true, quote='', escape='')"
    with duckdb.connect(":memory:") as db:
        db.execute(f"CREATE TEMP TABLE active AS SELECT DISTINCT concept_id::BIGINT id FROM {csv_reader} "
                   "WHERE vocabulary_id='SNOMED' AND coalesce(invalid_reason,'')=''", [concepts])
        expected_ids = db.sql("SELECT id FROM active ORDER BY id").fetchnumpy()["id"]
        pairs = db.execute(f"SELECT DISTINCT p.id AS parent, c.id AS child FROM {csv_reader} AS r "
                           "JOIN active p ON p.id=r.concept_id_2::BIGINT JOIN active c ON c.id=r.concept_id_1::BIGINT "
                           "WHERE r.relationship_id='Is a' AND coalesce(r.invalid_reason,'')='' ORDER BY 1, 2",
                           [relationships]).fetchnumpy()
    ids = np.asarray(graph.full_graph.nodes(), dtype=np.int64)
    pairs_in_graph = ids[np.asarray(graph.full_graph.edge_list(), dtype=np.int64)]
    ordered = pairs_in_graph[np.lexsort((pairs_in_graph[:, 1], pairs_in_graph[:, 0]))]
    np.testing.assert_array_equal(np.sort(ids), expected_ids)
    np.testing.assert_array_equal(ordered, np.column_stack((pairs["parent"], pairs["child"])))
    file = tmp_path_factory.mktemp("snomed") / "graph.pkl"
    graph.save(file)
    return file, expected_ids


@pytest.mark.parametrize("method", ("sampled", "landmark", *METHODS))
def test_export_on_supplied_snomed(method, supplied_taxonomy, tmp_path):
    if method in ("deepwalk", "node2vec"):
        pytest.importorskip("gensim")
    graph_file, expected_ids = supplied_taxonomy
    geometries = ("hyperbolic", "euclidean") if method in ("sampled", "landmark") else ("euclidean",)
    args = Args(graph_file=str(graph_file), graph_id="full", output_directory=str(tmp_path), experiment_id=method,
                training_method=method, geometry="both" if len(geometries) == 2 else geometries[0],
                directed=True, embedding_dim=8, epochs=1, burn_in=False, negative_samples=4, learning_rate=0.1,
                landmarks=24, svd_iterations=2, spectral_tolerance=0.0001, spectral_iterations=1000,
                walks_per_node=1, walk_length=4, walk_window=2, walk_epochs=1, inout_q=0.5,
                gae_epochs=2, gae_hidden=12, gae_edge_budget=8192, eval_interval=0, write_tsv=False, threads=2)
    train(args)
    for geometry in geometries:
        table = load_embedding(output_folder(args, geometry) / "embedding.pt", geometry=geometry)
        ids = table["concept_ids"].numpy()
        points = table["embeddings"]
        np.testing.assert_array_equal(np.sort(ids), expected_ids)
        assert points.shape == (len(expected_ids), args.embedding_dim) and torch.isfinite(points).all()
        assert ids[table["root_index"]] == args.root_node_label and points.count_nonzero() > 0
