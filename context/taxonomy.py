from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components


def _scan(path):
    return pl.scan_csv(Path(path), separator="\t", quote_char=None, infer_schema_length=0)


def _concept_id(value):
    number = int(value)
    if not isinstance(value, str) and number != value:
        raise ValueError(f"Expected an integer concept ID, got {value!r}")
    return number


def _ancestor_rows(adjacency, seeds):
    parents = adjacency.T.tocsr()
    node_count = parents.shape[0]
    indices = np.concatenate((parents.indices, seeds.astype(parents.indices.dtype)))
    offsets = np.append(parents.indptr, len(indices)).astype(parents.indptr.dtype)
    searchable = csr_matrix((np.ones(len(indices), dtype=bool), indices, offsets),
                            shape=(node_count + 1, node_count + 1))
    return breadth_first_order(searchable, node_count, return_predecessors=False)[1:]


def snomed_taxonomy(concept_file, relationship_file, concept_ids, root, include_all=False):
    if root is not None:
        root = _concept_id(root)
    elif not include_all:
        raise ValueError("A root is required to extract a subtree")
    active = pl.col("invalid_reason").fill_null("") == ""
    concepts = (_scan(concept_file).filter((pl.col("vocabulary_id") == "SNOMED") &
                (pl.col("standard_concept") == "S") & active)
                .select(pl.col("concept_id").cast(pl.Int64)).unique().sort("concept_id").collect(streaming=True))
    valid_ids = concepts["concept_id"].to_numpy()
    root_row = int(valid_ids.searchsorted(root)) if root is not None else None
    if not len(valid_ids) or (root is not None and (root_row == len(valid_ids) or valid_ids[root_row] != root)):
        raise ValueError("The requested taxonomy root is not an active standard SNOMED concept")
    relationships = (_scan(relationship_file).filter((pl.col("relationship_id") == "Is a") & active)
                     .select(pl.col("concept_id_2").cast(pl.Int64).alias("parent"),
                             pl.col("concept_id_1").cast(pl.Int64).alias("child")).drop_nulls()
                     .collect(streaming=True))
    parent_ids = relationships["parent"].to_numpy()
    child_ids = relationships["child"].to_numpy()
    parent_rows = valid_ids.searchsorted(parent_ids).clip(max=len(valid_ids) - 1)
    child_rows = valid_ids.searchsorted(child_ids).clip(max=len(valid_ids) - 1)
    standard_edge = (valid_ids[parent_rows] == parent_ids) & (valid_ids[child_rows] == child_ids)
    adjacency = csr_matrix((np.ones(standard_edge.sum(), dtype=bool),
                           (parent_rows[standard_edge], child_rows[standard_edge])),
                           shape=(len(valid_ids), len(valid_ids)))
    below_root = (np.arange(len(valid_ids)) if include_all else
                  np.sort(breadth_first_order(adjacency, root_row, return_predecessors=False)))
    if concept_ids is None:
        retained = covered = below_root
        requested_count = len(covered)
        outside_root = nonstandard = 0
    else:
        requested = np.unique(np.fromiter(map(_concept_id, concept_ids), dtype=np.int64))
        requested = requested[requested > 0]
        standard_ids = np.intersect1d(valid_ids, requested, assume_unique=True)
        covered = np.intersect1d(below_root, valid_ids.searchsorted(standard_ids), assume_unique=True)
        if not len(covered):
            raise ValueError("None of the requested concepts is in the specified rooted taxonomy")
        retained = below_root if len(covered) == len(below_root) else np.intersect1d(
            below_root, _ancestor_rows(adjacency, covered), assume_unique=True)
        requested_count = len(requested)
        outside_root = len(standard_ids) - len(covered)
        nonstandard = requested_count - len(standard_ids)
    subtree = adjacency if len(retained) == len(valid_ids) else adjacency[retained][:, retained]
    if subtree.diagonal().any() or connected_components(subtree, directed=True, connection="strong",
                                                        return_labels=False) != len(retained):
        raise ValueError("Cycle in requested SNOMED is-a taxonomy")
    parents, children = subtree.nonzero()
    retained_ids = valid_ids[retained]
    parent_counts = np.bincount(children, minlength=len(retained))
    isolated = (parent_counts == 0) & (np.diff(subtree.indptr) == 0)
    summary = {"root": root, "requested": requested_count, "covered": len(covered),
               "excluded_nonstandard_or_other_vocabulary": nonstandard, "excluded_outside_root": outside_root,
               "nodes": len(retained), "edges": len(parents), "intermediate_nodes": len(retained) - len(covered),
               "multiple_parent_nodes": int((parent_counts > 1).sum()), "isolated_nodes": int(isolated.sum()),
               "scope": "all" if include_all else ("subtree" if concept_ids is None else "seeds"),
               "edge_source": "CONCEPT_RELATIONSHIP, active Is a with standard SNOMED endpoints"}
    return pl.DataFrame({"parent": retained_ids[parents], "child": retained_ids[children]}), retained_ids, summary


def rooted_snomed_edges(concept_file, relationship_file, concept_ids, root=441840):
    """Return parent-to-child edges, retained concept IDs and coverage for the rooted taxonomy."""
    table, nodes, summary = snomed_taxonomy(concept_file, relationship_file, concept_ids, root)
    return table.rows(), nodes.tolist(), summary
