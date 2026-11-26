import json
from pathlib import Path

import pytest

from context import panel
from context.classical import METHODS
from context.export import load_embedding
from context.graph import Graph


@pytest.fixture
def template(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "schema": "context-representation-panel/1",
        "shared": {"embedding_dim": 4, "root_node_label": 1, "directed": True,
                   "eval_interval": 0, "write_tsv": False, "threads": 1},
        "arms": [{"name": "ids", "settings": {"training_method": "random", "geometry": "euclidean"}},
                 {"name": "ancestors", "settings": {"training_method": "ancestor", "geometry": "euclidean"}}],
        "permutations": [{"arm": "ancestors", "matching": "degree", "seed": 7}]}))
    return path


@pytest.fixture
def vocabulary(tmp_path):
    (tmp_path / "CONCEPT.csv").write_text("concept_id\tvocabulary_id\tstandard_concept\tinvalid_reason\n"
                                         + "".join(f"{i}\tSNOMED\tS\t\n" for i in range(1, 7)))
    (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(
        "concept_id_1\tconcept_id_2\trelationship_id\tinvalid_reason\n"
        "2\t1\tIs a\t\n3\t1\tIs a\t\n4\t2\tIs a\t\n4\t3\tIs a\t\n5\t2\tIs a\t\n")
    return tmp_path


def test_packaged_template_contains_every_method():
    settings = [arm["settings"] for arm in panel.plan()["arms"]]
    assert {arm["training_method"] for arm in settings} == {"sampled", "landmark", *METHODS}
    assert {arm["training_method"] for arm in settings if arm["geometry"] == "both"} == {"sampled", "landmark"}


@pytest.mark.parametrize("change", ["repeated_arm", "reserved_name", "mixed_partition", "mixed_width", "duplicate_shuffle", "owned_path"])
def test_incompatible_panel_is_rejected_before_training(template, change):
    spec = json.loads(template.read_text())
    second = spec["arms"][1]
    if change == "repeated_arm":
        second["name"] = "IDs"
    elif change == "reserved_name":
        second["name"] = "panel.json"
    elif change == "mixed_partition":
        second["settings"]["split_seed"] = 99
    elif change == "mixed_width":
        second["settings"]["embedding_dim"] = 6
    elif change == "duplicate_shuffle":
        spec["permutations"] *= 2
    else:
        spec["shared"]["output_directory"] = "elsewhere"
    template.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        panel.plan(template)


@pytest.mark.parametrize("settings", [
    {"retrieval_direction": "children"}, {"eval_queries": 1}, {"query_seed": 99},
])
def test_scored_arms_must_use_matched_retrieval_queries(template, settings):
    spec = json.loads(template.read_text())
    spec["shared"].update(validation_fraction=0.1, eval_interval=1)
    spec["arms"][0]["settings"]["retrieval_direction"] = "parents"
    spec["arms"][1]["settings"].update({"retrieval_direction": "parents", **settings})
    template.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="retrieval"):
        panel.plan(template)


@pytest.mark.parametrize("eval_interval, eval_queries", [(1, 0), (0, 2)])
def test_unused_query_seed_can_differ(template, eval_interval, eval_queries):
    spec = json.loads(template.read_text())
    spec["shared"].update(eval_interval=eval_interval, eval_queries=eval_queries)
    spec["arms"][1]["settings"]["query_seed"] = 99
    template.write_text(json.dumps(spec))
    assert panel.plan(template) == spec


def test_raw_seed_paths_resume_and_shuffled_artifact_integrity(template, vocabulary, tmp_path, monkeypatch):
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("conceptId\n4\n")
    output = tmp_path / "prepared"
    fit = panel.train
    def interrupt(args):
        if args.experiment_id == "ancestors":
            raise RuntimeError("interrupted")
        return fit(args)
    with monkeypatch.context() as patch:
        patch.setattr(panel, "train", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            panel.prepare(template, None, output, vocabulary=vocabulary, concept_ids=seeds)
    first_table = output / "ids/random/euclidean/embedding.pt"
    first_time = first_table.stat().st_mtime_ns
    assert not (output / "panel.json").exists()
    result = panel.prepare(template, None, output, vocabulary=vocabulary, concept_ids=seeds)
    assert result == panel.prepare(template, None, output, vocabulary=vocabulary, concept_ids=seeds)
    assert first_time == first_table.stat().st_mtime_ns
    assert result["taxonomy"]["nodes"] == 4 and result["taxonomy"]["edges"] == 4
    assert set(result["vocabulary_inputs"]["files"]) == {"concepts", "relationships", "seeds"}
    assert set(load_embedding(first_table)["concept_ids"].tolist()) == {1, 2, 3, 4}
    assert len(result["controls"]) == 1 and result["controls"][0]["seed"] == 7
    shuffle = output / Path(result["controls"][0]["path"]).parent / "permutation.npy"
    shuffle.write_bytes(shuffle.read_bytes() + b"x")
    with pytest.raises(ValueError, match="permutation"):
        panel.prepare(template, None, output, vocabulary=vocabulary, concept_ids=seeds)
    seeds.write_text("conceptId\n5\n")
    with pytest.raises(ValueError, match="Graph inputs changed"):
        panel.prepare(template, None, output, vocabulary=vocabulary, concept_ids=seeds)


def test_saved_graph_and_template_identity(template, vocabulary, tmp_path):
    all_output = tmp_path / "all"
    panel.prepare(template, None, all_output, vocabulary=vocabulary, taxonomy_scope="all")
    graph = all_output / "taxonomy.pkl"
    assert Graph.load(graph).full_graph.num_nodes() == 6
    output = tmp_path / "from-cache"
    receipt = panel.prepare(template, graph, output)
    assert receipt == json.loads((output / "panel.json").read_text())
    assert receipt["graph_sha256"] and len(receipt["representations"]) == 2
    template.write_text(template.read_text() + "\n")
    with pytest.raises(ValueError, match="different panel"):
        panel.prepare(template, graph, output)
