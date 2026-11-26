import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
import json
from pathlib import Path

from filelock import FileLock

from context.args import Args
from context.controls import permute_embedding
from context.export import file_sha, load_embedding
from context.train import _validate, graph_for_run, output_folder, train


def _read_template(template):
    source = files("context").joinpath("taxonomy-panel.json") if template is None else Path(template)
    raw = source.read_bytes()
    spec = json.loads(raw)
    if (not isinstance(spec, dict) or spec.get("schema") != "context-representation-panel/1"
            or not isinstance(spec.get("arms"), list) or not spec["arms"] or not isinstance(spec.get("shared"), dict)):
        raise ValueError("Expected a context-representation-panel/1 template with at least one arm")
    shared = spec["shared"]
    managed = {"experiment_id", "output_directory", "graph_file", "resume", "refit_from",
               "concept_id_file", "concept_map", "relationship_file_path", "hierarchy_file_path",
               "taxonomy_scope", "snomed_concepts"}
    used = {"panel.json", "panel.identity", "taxonomy.pkl"}
    options = {}
    comparison = None
    for arm in spec["arms"]:
        name, settings = arm["name"], arm["settings"]
        if (not isinstance(name, str) or not name or name.startswith(".") or "/" in name or "\\" in name
                or name.casefold() in used):
            raise ValueError(f"Provide a distinct directory name for each arm: {name!r}")
        used.add(name.casefold())
        if not isinstance(settings, dict) or shared.keys() & settings.keys() or managed & (shared.keys() | settings.keys()):
            raise ValueError("Panel inputs, output paths and shared settings cannot be overridden by an arm")
        args = Args(**shared, **settings)
        _validate(args)
        graph_settings = (args.graph_id, args.directed, args.root_node_label, args.embedding_dim,
                          args.validation_fraction, args.split_seed, args.split_policy)
        if comparison is not None and graph_settings != comparison:
            raise ValueError("All panel arms must use the same graph, direction, root, width and partition settings")
        comparison = graph_settings
        options[name] = args
    permutations = set()
    for control in spec.get("permutations", []):
        arm, matching, seed = control["arm"], control["matching"], control["seed"]
        if arm not in options or matching not in ("degree", "none") or type(seed) is not int or seed < 0:
            raise ValueError("Invalid permutation source, matching scheme or seed")
        identity = (arm, matching, seed)
        if identity in permutations:
            raise ValueError("Duplicate permutation in panel")
        permutations.add(identity)
    return spec, options, hashlib.sha256(raw).hexdigest()


def plan(template=None):
    return _read_template(template)[0]


def _write_text(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _permutation(source, output, graph, arm, geometry, source_sha, control):
    matching, seed = control["matching"], control["seed"]
    directory = source.parent / f"permutation-{matching}-{seed}"
    payload = directory / "embedding.pt"
    if directory.exists():
        metadata = load_embedding(payload)["metadata"]
        expected = (source_sha, seed, matching, file_sha(directory / "permutation.npy"))
        if tuple(metadata[key] for key in ("source_sha256", "seed", "matching", "permutation_sha256")) != expected:
            raise ValueError(f"Existing permutation differs from the requested source or settings: {directory}")
    else:
        metadata = permute_embedding(source, directory, matching=matching, seed=seed, graph_file=graph)
    return {"arm": arm, "geometry": geometry, "matching": matching, "seed": seed,
            "rows_with_changed_vector": metadata["rows_with_changed_vector"],
            "path": str(payload.relative_to(output)), "sha256": file_sha(payload)}


def prepare(template, graph, output, *, vocabulary=None, concept_ids=None, taxonomy_scope=None, snomed_concepts=None):
    spec, arms, template_sha = _read_template(template)
    if (graph is None) == (vocabulary is None):
        raise ValueError("Supply either a trusted graph or an OMOP vocabulary directory")
    if graph is not None and any(item is not None for item in (concept_ids, taxonomy_scope, snomed_concepts)):
        raise ValueError("Vocabulary extraction settings require a vocabulary directory")
    if any(opts.training_method in ("deepwalk", "node2vec") for opts in arms.values()):
        try:
            version("gensim")
        except PackageNotFoundError as error:
            raise ImportError("This panel uses walk methods. Install the walks extra with uv sync --extra walks.") from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".panel.lock"):
        key = output / "panel.identity"
        if not key.exists() and any((output / name).exists() for name in (*arms, "panel.json")):
            raise ValueError("Use a new output directory or resume an existing identified panel")
        inputs = None
        extraction = None
        policy = {}
        if vocabulary is not None:
            scope = taxonomy_scope or ("subtree" if concept_ids is None else "seeds")
            if (scope == "seeds") != (concept_ids is not None):
                raise ValueError("The seeds scope requires a concept ID file. Other scopes take no seed file.")
            if scope not in ("all", "subtree", "seeds"):
                raise ValueError("Unknown taxonomy scope")
            policy = {"taxonomy_scope": scope, "snomed_concepts": snomed_concepts or "standard"}
            vocabulary = Path(vocabulary)
            options = replace(next(iter(arms.values())), **policy,
                              concept_map=str(vocabulary / "CONCEPT.csv"),
                              relationship_file_path=str(vocabulary / "CONCEPT_RELATIONSHIP.csv"),
                              concept_id_file=str(concept_ids) if concept_ids is not None else Args.concept_id_file)
            graph = output / "taxonomy.pkl"
            cached_graph = graph_for_run(options, graph)
            extraction, inputs = cached_graph.extraction, cached_graph.source_inputs
            del cached_graph
        graph = Path(graph)
        identity = {"template_sha256": template_sha, "graph_sha256": file_sha(graph), "package_version": version("context")}
        if inputs is not None:
            identity["vocabulary_inputs"] = inputs
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key.exists() and key.read_text() != signature:
            raise ValueError("This directory belongs to a different panel, graph or package version")
        _write_text(key, signature)
        result = {**identity, "settings": spec, "representations": [], "controls": []}
        if extraction is not None:
            result["taxonomy"] = extraction
        split = None
        for name, args in arms.items():
            opts = replace(args, graph_file=str(graph), output_directory=str(output), experiment_id=name, resume=True, **policy)
            train(opts)
            geometries = ("euclidean", "hyperbolic") if opts.geometry == "both" else (opts.geometry,)
            for geometry in geometries:
                folder = output_folder(opts, geometry)
                fit = json.loads((folder / "training.json").read_text())
                if fit["graph_sha256"] != identity["graph_sha256"]:
                    raise ValueError("The input graph changed during preparation")
                if opts.validation_fraction:
                    if split is not None and fit["split_sha256"] != split:
                        raise ValueError("Training, validation or test partitions differ across arms")
                    split = fit["split_sha256"]
                source = folder / "embedding.pt"
                source_sha = fit["output_sha256"]["embedding.pt"]
                result["representations"].append({"arm": name, "method": opts.training_method, "geometry": geometry,
                         "nodes": fit["nodes"], "dimensions": opts.embedding_dim, "split_sha256": fit.get("split_sha256"),
                         "path": str(source.relative_to(output)), "sha256": source_sha})
                for control in spec.get("permutations", []):
                    if control["arm"] == name:
                        result["controls"].append(_permutation(source, output, graph, name, geometry, source_sha, control))
        _write_text(output / "panel.json", json.dumps(result, indent=2) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description="Prepare matched concept embeddings from one taxonomy.")
    parser.add_argument("--template", type=Path, help="Panel JSON. Defaults to the bundled twelve-method template.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--graph", type=Path, help="Trusted context graph pickle")
    inputs.add_argument("--vocabulary", type=Path, help="Directory containing CONCEPT.csv and CONCEPT_RELATIONSHIP.csv")
    parser.add_argument("--concept-ids", type=Path, help="CSV with a conceptId column for seed-to-root extraction")
    parser.add_argument("--taxonomy-scope", choices=("all", "subtree", "seeds"))
    parser.add_argument("--snomed-concepts", choices=("active", "standard"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the template without reading a graph")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(plan(args.template), indent=2))
    elif args.output is None or (args.graph is None and args.vocabulary is None):
        parser.error("Provide --output and either --graph or --vocabulary")
    else:
        prepare(args.template, args.graph, args.output, vocabulary=args.vocabulary, concept_ids=args.concept_ids,
                taxonomy_scope=args.taxonomy_scope, snomed_concepts=args.snomed_concepts)


if __name__ == "__main__":
    main()
