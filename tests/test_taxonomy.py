import sqlite3
from zipfile import ZipFile

import duckdb
import pytest

from context.args import Args
from context.graph import Graph
from context.train import build_graph


def test_typed_diamond_keeps_all_paths_and_excludes_other_relations(tmp_path):
    concepts = tmp_path / "CONCEPT.csv"
    concepts.write_text("concept_id\tconcept_name\tvocabulary_id\tstandard_concept\tinvalid_reason\n"
                        + "".join(f"{i}\tc{i}\tSNOMED\tS\t\n" for i in (1, 2, 3, 4, 5, 6))
                        + "7\tc7\tOTHER\tS\t\n8\tc8\tSNOMED\tS\tD\n")
    relationships = tmp_path / "CONCEPT_RELATIONSHIP.csv"
    relationships.write_text("concept_id_1\tconcept_id_2\trelationship_id\tinvalid_reason\n"
                             "2\t1\tIs a\t\n3\t1\tIs a\t\n4\t2\tIs a\t\n4\t3\tIs a\t\n"
                             "4\t3\tIs a\t\n4\t1\tIs a\t\n5\t1\tAssociated with\t\n"
                             "6\t1\tIs a\tD\n7\t1\tIs a\t\n8\t1\tIs a\t\n")
    ids = tmp_path / "ids.csv"
    ids.write_text("conceptId\n4\n5\n6\n7\n8\n")
    g = build_graph(Args(concept_id_file=ids, concept_map=concepts, relationship_file_path=relationships,
                         root_node_label=1))
    graph = g.subgraph["ancestral"]
    assert {(graph[s], graph[t]) for s, t in graph.edge_list()} == {(1, 2), (1, 3), (1, 4), (2, 4), (3, 4)}
    assert g.extraction["covered"] == 1 and g.extraction["multiple_parent_nodes"] == 1
    g.save(tmp_path / "graph.pkl")
    assert Graph.load(tmp_path / "graph.pkl").extraction == g.extraction
    with pytest.raises(ValueError, match="integer concept ID"):
        Graph.from_snomed_taxonomy(concepts, relationships, [4.5], root=1)
    with relationships.open("a") as stream:
        stream.write("2\t4\tIs a\t\n")
    with pytest.raises(ValueError, match="[Cc]ycle"):
        Graph.from_snomed_taxonomy(concepts, relationships, [4], root=1)


@pytest.mark.parametrize("engine", ["duckdb", "sqlite"])
def test_plp_dictionary_reads_mapped_ids_and_fails_on_missing_source(tmp_path, engine):
    path = tmp_path / ("feature dict." + engine)
    conn = duckdb.connect(str(path)) if engine == "duckdb" else sqlite3.connect(path)
    conn.execute("create table covariateRef(conceptId int, covariateId bigint)")
    conn.executemany("insert into covariateRef values (?, ?)", [(42, 42001), (42, 42002), (0, 8), (None, 9)])
    conn.commit()
    conn.close()
    with ZipFile(tmp_path / "covariates", "w") as archive:
        archive.write(path, "nested/" + path.name)
    assert Graph.get_concept_set([tmp_path]) == [42, 441840]
    with pytest.raises(ValueError, match="PLP"):
        Graph.get_concept_set([tmp_path, tmp_path / "absent"])
    with pytest.warns(RuntimeWarning):
        assert Graph.get_concept_set([tmp_path, tmp_path / "absent"], allow_partial=True, root_concept_id=None) == [42]
