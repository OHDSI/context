"""Extract a single rooted SNOMED is-a taxonomy from OMOP vocabulary exports."""
from collections import defaultdict
from pathlib import Path

import polars as pl


def _scan(path):
    return pl.scan_csv(Path(path), separator="\t", quote_char=None, infer_schema_length=0)


def _reach(seeds, neighbours):
    visited = set(seeds)
    pending = list(visited)
    while pending:
        node = pending.pop()
        for other in neighbours.get(node, ()):
            if other not in visited:
                visited.add(other)
                pending.append(other)
    return visited


def _concept_id(value):
    number = int(value)
    if not isinstance(value, str) and number != value:
        raise ValueError(f"Expected an integer concept ID, got {value!r}")
    return number


def rooted_snomed_edges(concept_file, relationship_file, concept_ids, root=441840):
    """Return parent-child edges, retained identifiers and aggregate coverage.

    Is a in OMOP CONCEPT_RELATIONSHIP points from child to parent. Retain every
    requested-to-root path. A closure table alone cannot establish edge type.
    """
    root = _concept_id(root)
    requested = {i for i in map(_concept_id, concept_ids) if i > 0}
    active = pl.col("invalid_reason").fill_null("") == ""
    concepts = _scan(concept_file).filter(
        (pl.col("vocabulary_id") == "SNOMED") & (pl.col("standard_concept") == "S") & active
    ).select(pl.col("concept_id").cast(pl.Int64)).collect()
    valid = set(concepts["concept_id"])
    if root not in valid:
        raise ValueError("The requested taxonomy root is not an active standard SNOMED concept")
    rel = (_scan(relationship_file).filter((pl.col("relationship_id") == "Is a") & active)
           .select(pl.col("concept_id_1").cast(pl.Int64).alias("child"),
                   pl.col("concept_id_2").cast(pl.Int64).alias("parent")).collect())
    parents, children = defaultdict(set), defaultdict(set)
    for child, parent in rel.iter_rows():
        if child in valid and parent in valid:
            parents[child].add(parent)
            children[parent].add(child)
    below = _reach({root}, children)
    accepted = requested & below
    if not accepted:
        raise ValueError("None of the requested concepts is in the specified rooted taxonomy")
    upward = _reach(accepted, parents)
    nodes = below & upward
    edges = sorted((par, ch) for ch in nodes for par in parents.get(ch, ()) if par in nodes)
    # Kahn traversal checks the induced taxonomy, including self-loops.
    degree = {ch: len(parents.get(ch, set()) & nodes) for ch in nodes}
    pending = [node for node, count in degree.items() if count == 0]
    if set(pending) != {root}:
        raise ValueError("Rooted taxonomy has a cycle or an unexpected root")
    seen = 0
    while pending:
        parent = pending.pop()
        seen += 1
        for child in children.get(parent, ()):
            if child in nodes:
                degree[child] -= 1
                if degree[child] == 0:
                    pending.append(child)
    if seen != len(nodes):
        raise ValueError("Cycle in requested SNOMED is-a taxonomy")
    summary = {"root": root, "requested": len(requested), "covered": len(accepted),
               "excluded_nonstandard_or_other_vocabulary": len(requested - valid),
               "excluded_outside_root": len((requested & valid) - below),
               "nodes": len(nodes), "edges": len(edges),
               "intermediate_nodes": len(nodes - accepted),
               "multiple_parent_nodes": sum(len(parents.get(n, set()) & nodes) > 1 for n in nodes),
               "edge_source": "CONCEPT_RELATIONSHIP, active Is a with standard SNOMED endpoints"}
    return edges, sorted(nodes), summary
