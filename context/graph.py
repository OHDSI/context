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

from context.taxonomy import rooted_snomed_edges

class Graph:
    def __init__(self, hierarchy):
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
        # todo: a subgraph that only contains the nodes specified in concept_ids and must create new edges between them.
        #  It may be best to use ancestral subgraph as basis.
        return None

    def ancestral_subgraph(self, nodes, name="ancestral"):
        # One reverse traversal from all requested seeds preserves every parent.
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
        """Build the ancestral subgraph from one-step entries in CONCEPT_ANCESTOR."""
        # A direct parent can also have a longer alternative path in a DAG.
        edges = (pl.scan_csv(ancestor_file, separator="\t", quote_char=None)
                 .filter((pl.col("min_levels_of_separation") == 1) &
                         (pl.col("ancestor_concept_id") != pl.col("descendant_concept_id")))
                 .select("ancestor_concept_id", "descendant_concept_id").unique()
                 .sort(["ancestor_concept_id", "descendant_concept_id"]).collect())
        graph = cls(edges.with_columns(pl.lit("subsumes").alias("relation")))
        graph.ancestral_subgraph(concept_ids, name)
        return graph

    @classmethod
    def from_snomed_taxonomy(cls, concept_file, relationship_file, concept_ids, root=441840, name="ancestral"):
        """Build a rooted directed acyclic graph of explicit SNOMED is-a statements."""
        edges, nodes, summary = rooted_snomed_edges(concept_file, relationship_file, concept_ids, root)
        table = pl.DataFrame(edges, schema={"parent": pl.Int64, "child": pl.Int64}, orient="row")
        graph = cls(table.with_columns(pl.lit("subsumes").alias("relation")))
        if not edges:
            graph.full_graph.add_node(nodes[0])
        graph.subgraph[name] = graph.full_graph.copy()
        graph.extraction = summary
        return graph

    @staticmethod
    def get_concept_set(data_object_dirs, output_folder=None, allow_partial=False, root_concept_id=441840):
        """Read mapped concepts from every supplied PLP object, failing on incomplete input by default."""
        all_concept_ids = set()
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
        """Use mapped PLP concepts and all their ancestors."""
        return self.ancestral_subgraph(self.get_concept_set(data_object_dirs), name=name)

    @staticmethod
    def _process_data_object(data_object_dir):
        zip_path = Path(data_object_dir) / "covariates"
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        with tempfile.TemporaryDirectory() as tmpdirname, ZipFile(zip_path) as archive:
            members = [m for m in archive.infolist() if Path(m.filename).suffix in {".sqlite", ".duckdb"}
                       and not m.is_dir()]
            if len(members) != 1:
                raise ValueError("Expected one SQLite or DuckDB database in the covariate archive")
            target = Path(tmpdirname) / ("covariates" + Path(members[0].filename).suffix)
            with archive.open(members[0]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if target.suffix == ".sqlite":
                conn = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)
            else:
                conn = duckdb.connect(str(target), read_only=True)
            try:
                query = 'SELECT DISTINCT conceptId AS "conceptId" FROM covariateRef'
                # Materialize before the connection and temporary directory go away.
                if target.suffix == ".duckdb":
                    return conn.sql(query).pl().lazy()
                return pl.read_database(query, conn).lazy()
            finally:
                conn.close()

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

    def save(self, file_path):
        data = {
            "extraction": getattr(self, "extraction", None),
            "full_graph": {
                "nodes": [self.full_graph[node_idx] for node_idx in self.full_graph.node_indexes()],
                "edges": [
                    (self.full_graph[source], self.full_graph[target], edge_data)
                    for source, target, edge_data in self.full_graph.weighted_edge_list()
                ],
            },
            "subgraphs": {
                name: {
                    "nodes": [sub[node_idx] for node_idx in sub.node_indexes()],
                    "edges": [
                        (sub[source], sub[target], edge_data)
                        for source, target, edge_data in sub.weighted_edge_list()
                    ],
                }
                for name, sub in self.subgraph.items()
            },
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        inst = cls.__new__(cls)
        if data.get("extraction") is not None:
            inst.extraction = data["extraction"]

        inst.full_graph = rx.PyDiGraph()
        node_map = {}
        for node in data["full_graph"]["nodes"]:
            new_index = inst.full_graph.add_node(node)
            node_map[node] = new_index
        for src, tgt, edge_data in data["full_graph"]["edges"]:
            inst.full_graph.add_edge(node_map[src], node_map[tgt], edge_data)

        inst.subgraph = {}
        for name, sgdata in data["subgraphs"].items():
            sub = rx.PyDiGraph()
            sub_node_map = {}
            for node in sgdata["nodes"]:
                new_index = sub.add_node(node)
                sub_node_map[node] = new_index
            for src, tgt, edge_data in sgdata["edges"]:
                sub.add_edge(sub_node_map[src], sub_node_map[tgt], edge_data)
            inst.subgraph[name] = sub

        return inst
