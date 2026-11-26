from itertools import islice, repeat

import numpy as np
import rustworkx as rx
import time
import pickle
import sqlite3
import warnings
import tempfile
import polars as pl
import duckdb
import shutil
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile

from context.taxonomy import snomed_taxonomy

class Graph:
    def __init__(self, hierarchy, *, nodes=()):
        """
        :param hierarchy: full hierarchy edge list (e.g. SNOMED's
        CONCEPT_ANCESTOR.csv)
        """
        self.full_graph = rx.PyDiGraph()
        self.subgraph = {}

        start = time.time()

        existing_nodes = {}
        for source, target, edge_data in hierarchy.iter_rows(named=False):
            if source not in existing_nodes:
                source_index = self.full_graph.add_node(source)
                existing_nodes[source] = source_index
            else:
                source_index = existing_nodes[source]
            if target not in existing_nodes:
                target_index = self.full_graph.add_node(target)
                existing_nodes[target] = target_index
            else:
                target_index = existing_nodes[target]
            self.full_graph.add_edge(source_index, target_index, edge_data)

        self.full_graph.add_nodes_from([node for node in nodes if node not in existing_nodes])
        delta = time.time() - start
        print(f"Processed edge list in {delta:.2f} seconds")
        print(repr(self))

    def __repr__(self):
        subgraph_info = ", ".join(
            f"{name}: ({subgraph.num_nodes()} nodes, "
            f"{subgraph.num_edges()} edges)"
            for name, subgraph in self.subgraph.items()
        )
        return (
            f"Directed graph has {self.full_graph.num_nodes()} nodes "
            f"and {self.full_graph.num_edges()} edges; "
            f"Subgraphs: [{subgraph_info}]"
        )

    def _to_networkx(self, rustworkx_graph):
        import networkx as nx
        start = time.time()

        nx_graph = nx.DiGraph()
        for node_index in rustworkx_graph.node_indexes():
            nx_graph.add_node(rustworkx_graph[node_index])
        for source, target, edge_data in rustworkx_graph.weighted_edge_list():
            nx_graph.add_edge(rustworkx_graph[source], rustworkx_graph[target], data=edge_data)

        delta = time.time() - start
        print(f"Converted to networkx graph in {delta:.2f} seconds")
        return nx_graph

    def _to_igraph(self, rustworkx_graph):
        import igraph as ig
        start = time.time()

        igraph_graph = ig.Graph(directed=True)
        node_labels = [rustworkx_graph[node_index] for node_index in rustworkx_graph.node_indexes()]
        igraph_graph.add_vertices(len(node_labels))
        igraph_graph.vs['name'] = node_labels
        label_to_index = {label: index for index, label in enumerate(node_labels)}
        edges = [(label_to_index[rustworkx_graph[source]], label_to_index[rustworkx_graph[target]], edge_data)
                 for source, target, edge_data in rustworkx_graph.weighted_edge_list()]
        igraph_graph.add_edges([(source, target) for source, target, _ in edges])
        igraph_graph.es['weight'] = [edge_data for _, _, edge_data in edges]

        delta = time.time() - start
        print(f"Converted to igraph in {delta:.2f} seconds")
        return igraph_graph

    def _to_rustworkx(self, igraph_graph):
        start = time.time()

        rustworkx_graph = rx.PyDiGraph()
        for vertex in tqdm(igraph_graph.vs, desc="Converting nodes to rustworkx graph"):
            rustworkx_graph.add_node(vertex['name'])
        weights = igraph_graph.es["weight"] if "weight" in igraph_graph.edge_attributes() else [None] * igraph_graph.ecount()
        rustworkx_graph.add_edges_from([(s, t, w) for (s, t), w in zip(igraph_graph.get_edgelist(), weights)])

        delta = time.time() - start
        print(f"Converted to rustworkx in {delta:.2f} seconds")
        return rustworkx_graph

    def plot(self, graph_name="full", edge_labels=True):
        import matplotlib.pyplot as plt
        import networkx as nx
        if graph_name == "full":
            rustworkx_graph = self.full_graph
        elif graph_name in self.subgraph:
            rustworkx_graph = self.subgraph[graph_name]
        else:
            raise ValueError(f"Graph '{graph_name}' not found. Available options are 'full' and subgraphs: {list(self.subgraph.keys())}")

        nx_graph = self._to_networkx(rustworkx_graph)
        pos = nx.spring_layout(nx_graph, k=0.5, seed=42)
        nx.draw(nx_graph, pos, with_labels=True, node_size=250, node_color="lightblue", font_size=10)
        if edge_labels:
            edge_data = nx.get_edge_attributes(nx_graph, 'data')
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_data)
        plt.show()

    def selective_subgraph(self, concept_ids, name="selective"):
        concepts = set(concept_ids)
        graph = self.full_graph
        selected = rx.PyDiGraph(multigraph=False)
        rows = {index: selected.add_node(graph[index]) for index in graph.node_indices() if graph[index] in concepts}
        nearest = {}
        try:
            order = rx.topological_sort(graph)
        except rx.DAGHasCycle as error:
            raise ValueError("Selecting taxonomy ancestors requires an acyclic graph") from error
        for node in order:
            ancestors = nearest.pop(node, set())
            if node in rows:
                for ancestor in sorted(ancestors):
                    relation = graph.get_edge_data(ancestor, node) if graph.has_edge(ancestor, node) else "ancestor_path"
                    selected.add_edge(rows[ancestor], rows[node], relation)
                ancestors = {node}
            if ancestors:
                for child in graph.successor_indices(node):
                    nearest.setdefault(child, set()).update(ancestors)
        self.subgraph[name] = selected
        return selected

    def ancestral_subgraph(self, nodes, name="ancestral"):
        lookup = {self.full_graph[i]: i for i in self.full_graph.node_indexes()}
        selected = {lookup[n] for n in nodes if n in lookup}
        frontier = list(selected)
        while frontier:
            child = frontier.pop()
            for parent in self.full_graph.predecessor_indices(child):
                if parent not in selected:
                    selected.add(parent)
                    frontier.append(parent)
        result = self.full_graph.subgraph(sorted(selected))
        self.subgraph[name] = result
        return result

    @classmethod
    def from_omop(cls, ancestor_file, concept_ids, name="ancestral"):
        edges = (pl.scan_csv(ancestor_file, separator="\t", quote_char=None)
                 .filter((pl.col("min_levels_of_separation") == 1) &
                         (pl.col("ancestor_concept_id") != pl.col("descendant_concept_id")))
                 .select("ancestor_concept_id", "descendant_concept_id").unique()
                 .sort(["ancestor_concept_id", "descendant_concept_id"]).collect())
        graph = cls(edges.with_columns(pl.lit("subsumes").alias("relation")))
        graph.ancestral_subgraph(concept_ids, name)
        return graph

    @classmethod
    def from_snomed_taxonomy(cls, concept_file, relationship_file, concept_ids=None, root=441840, name="ancestral", *,
                             include_all=False, snomed_concepts="standard"):
        if include_all and concept_ids is not None:
            raise ValueError("Use concept seeds or request the entire SNOMED vocabulary")
        table, nodes, summary = snomed_taxonomy(concept_file, relationship_file, concept_ids, root, include_all, snomed_concepts)
        graph = cls(table.with_columns(pl.lit("subsumes").alias("relation")),
                    nodes=nodes.tolist() if summary["isolated_nodes"] else ())
        graph.subgraph[name] = graph.full_graph.copy()
        graph.extraction = summary
        return graph

    @staticmethod
    def get_concept_set(data_object_dirs, output_folder=None, allow_partial=False, root_concept_id=441840):
        all_concept_ids = set()
        if isinstance(data_object_dirs, (str, Path)):
            data_object_dirs = [data_object_dirs]
        if not data_object_dirs:
            raise ValueError("Supply at least one PLP data object")
        for dir_path in data_object_dirs:
            try:
                df = Graph._process_data_object(dir_path).collect()
                current = set()
                for cid in df["conceptId"].drop_nulls():
                    value = int(cid)
                    if value != float(cid) or value < 0:
                        raise ValueError("PLP concept IDs must be nonnegative integers")
                    current.add(value)
                all_concept_ids.update(current)
            except Exception as error:
                if not allow_partial:
                    raise ValueError(f"Could not read the PLP covariate dictionary in {dir_path}") from error
                warnings.warn(f"Skipping an unreadable PLP dictionary in {dir_path}: {error}", RuntimeWarning)
        if not (all_concept_ids - {0}):
            raise ValueError("PLP inputs contain no mapped concept IDs")

        # Remove the 0 value if present.
        all_concept_ids.discard(0)
        if root_concept_id is not None:
            all_concept_ids.add(root_concept_id)
        all_concept_ids = sorted(list(all_concept_ids))

        if output_folder is not None:
            result_df = pl.DataFrame({"conceptId": all_concept_ids}, schema={"conceptId": pl.Int64})
            output_folder_path = Path(output_folder)
            output_folder_path.mkdir(parents=True, exist_ok=True)
            output_csv = output_folder_path / "concept_ids.csv"
            result_df.write_csv(output_csv)
            print(f"Saved all concept ids to {output_csv}")
        return all_concept_ids

    def ancestral_subgraph_from_plp_data(self, data_object_dirs, name="ancestral"):
        return self.ancestral_subgraph(self.get_concept_set(data_object_dirs), name=name)

    @staticmethod
    def _process_data_object(data_object_dir):
        source_path = Path(data_object_dir)
        if source_path.is_dir():
            source_path /= "covariates"
        if source_path.suffix.lower() in (".sqlite", ".duckdb"):
            return Graph._read_dictionary(source_path)
        with tempfile.TemporaryDirectory() as temporary, ZipFile(source_path) as archive:
            members = [member for member in archive.infolist()
                       if Path(member.filename).suffix.lower() in {".sqlite", ".duckdb"} and not member.is_dir()]
            if len(members) != 1:
                raise ValueError("Expected one SQLite or DuckDB database in the covariate archive")
            target = Path(temporary) / ("covariates" + Path(members[0].filename).suffix.lower())
            with archive.open(members[0]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            return Graph._read_dictionary(target)

    @staticmethod
    def _read_dictionary(path):
        sqlite = path.suffix.lower() == ".sqlite"
        connection = (sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) if sqlite
                      else duckdb.connect(str(path), read_only=True))
        try:
            query = 'SELECT DISTINCT conceptId AS "conceptId" FROM covariateRef'
            data = pl.read_database(query, connection) if sqlite else connection.sql(query).pl()
            return data.lazy()
        finally:
            connection.close()

    def print_nodes(self, graph_name="full"):
        graph = (
            self.full_graph
            if graph_name == "full"
            else self.subgraph.get(graph_name)
        )
        if not graph:
            print(f"No graph named '{graph_name}' found.")
            return
        node_list = [str(node) for node in graph.nodes()]
        print(f"Nodes in {graph_name} graph ({graph.num_nodes()}): [{', '.join(node_list)}]")

    def find_root_node(self, graph_name="full"):
        if graph_name == "full":
            rustworkx_graph = self.full_graph
        elif graph_name in self.subgraph:
            rustworkx_graph = self.subgraph[graph_name]
        else:
            raise ValueError(
                f"Graph '{graph_name}' not found. Available options are 'full' and subgraphs: {list(self.subgraph.keys())}"
            )
        root_nodes = []
        for node_index in rustworkx_graph.node_indexes():
            if rustworkx_graph.in_degree(node_index) == 0 and (rustworkx_graph.out_degree(node_index) > 0 or rustworkx_graph.num_nodes() == 1):
                root_nodes.append(rustworkx_graph[node_index])
        if len(root_nodes) == 0:
            raise ValueError("No root nodes found.")
        elif len(root_nodes) > 1:
            raise ValueError(f"Multiple root nodes found: {', '.join(map(str, root_nodes))}")

        print(f"Root node: {root_nodes[0]}")
        return root_nodes[0]

    @staticmethod
    def _pack_graph(graph):
        indices = np.asarray(graph.node_indices())
        edges = np.asarray(graph.edge_list(), dtype=np.uint32).reshape(-1, 2)
        if len(indices) and indices[-1] != len(indices) - 1:
            edges = np.searchsorted(indices, edges).astype(np.uint32)
        payloads = list(graph.edges())
        packed = {"nodes": list(graph.nodes()), "endpoints": edges}
        common = payloads[0] if payloads else None
        if isinstance(common, (int, float, str, bytes, type(None))) and all(
            type(value) is type(common) and value == common for value in payloads
        ):
            packed["edge_value"] = common
        else:
            packed["edge_values"] = payloads
        return packed

    def save(self, file_path):
        full = self._pack_graph(self.full_graph)
        subgraphs = {}
        for name, graph in self.subgraph.items():
            packed = self._pack_graph(graph)
            same_graph = ("edge_value" in full and "edge_value" in packed and
                          type(full["edge_value"]) is type(packed["edge_value"]) and
                          full["edge_value"] == packed["edge_value"] and packed["nodes"] == full["nodes"] and
                          np.array_equal(packed["endpoints"], full["endpoints"]))
            subgraphs[name] = None if same_graph else packed
        data = {"format_version": 2, "extraction": getattr(self, "extraction", None),
                "source_inputs": getattr(self, "source_inputs", None), "full_graph": full, "subgraphs": subgraphs}
        destination = Path(file_path)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            path = Path(temporary) / "graph.pkl"
            with path.open("wb") as stream:
                pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
            path.replace(destination)

    @staticmethod
    def _unpack_graph(data):
        graph = rx.PyDiGraph()
        graph.add_nodes_from(data["nodes"])
        if "endpoints" in data:
            sources, targets = data["endpoints"].T
            values = repeat(data["edge_value"]) if "edge_value" in data else data["edge_values"]
            for start in range(0, len(sources), 65536):
                stop = start + 65536
                payloads = values if "edge_value" in data else values[start:stop]
                graph.add_edges_from(list(zip(sources[start:stop].tolist(), targets[start:stop].tolist(), payloads)))
        else:
            rows = {node: index for index, node in enumerate(data["nodes"])}
            edges = ((rows[source], rows[target], value) for source, target, value in data["edges"])
            while batch := list(islice(edges, 65536)):
                graph.add_edges_from(batch)
        return graph

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as stream:
            data = pickle.load(stream)
        if data.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported graph file version")
        instance = cls.__new__(cls)
        if data.get("extraction") is not None:
            instance.extraction = data["extraction"]
        instance.source_inputs = data.get("source_inputs")
        instance.full_graph = cls._unpack_graph(data["full_graph"])
        instance.subgraph = {name: instance.full_graph.copy() if graph is None else cls._unpack_graph(graph)
                             for name, graph in data["subgraphs"].items()}
        return instance
