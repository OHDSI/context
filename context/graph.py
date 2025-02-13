import rustworkx as rx
import networkx as nx
import igraph as ig
import time
import pickle
import matplotlib.pyplot as plt
import tempfile
import polars as pl
import duckdb
import shutil
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import quote

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
            f"{name}: ({len(subgraph.node_indexes())} nodes, "
            f"{len(subgraph.edge_list())} edges)"
            for name, subgraph in self.subgraph.items()
        )
        return (
            f"Directed graph has {len(self.full_graph.node_indexes())} nodes "
            f"and {len(self.full_graph.edge_list())} edges; "
            f"Subgraphs: [{subgraph_info}]"
        )

    def _to_networkx(self, rustworkx_graph):
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
        for edge in tqdm(igraph_graph.es, desc="Converting edges to rustworkx graph"):
            source_label = igraph_graph.vs[edge.source]['name']
            target_label = igraph_graph.vs[edge.target]['name']
            weight = edge['weight'] if 'weight' in edge.attributes() else None
            source_rustworkx_index = rustworkx_graph.nodes().index(source_label)
            target_rustworkx_index = rustworkx_graph.nodes().index(target_label)
            rustworkx_graph.add_edge(source_rustworkx_index, target_rustworkx_index, weight)

        delta = time.time() - start
        print(f"Converted to rustworkx in {delta:.2f} seconds")
        return rustworkx_graph

    def plot(self, graph_name="full", edge_labels=True):
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
        start_time = time.time()

        igraph_full = self._to_igraph(self.full_graph)
        valid_node_indices = []
        for v in tqdm(igraph_full.vs, desc=f"Collecting valid node indices for {name} subgraph"):
            if v['name'] in nodes:
                valid_node_indices.append(v.index)
        all_relevant_indices = set()
        for index in tqdm(valid_node_indices, desc=f"Finding ancestor nodes for {name} subgraph"):
            ancestors = igraph_full.subcomponent(index, mode="in")
            all_relevant_indices.update(ancestors)
        igraph_subgraph = igraph_full.induced_subgraph(list(all_relevant_indices))
        rustworkx_subgraph = self._to_rustworkx(igraph_subgraph)

        self.subgraph[name] = rustworkx_subgraph
        print(f"Created {name} subgraph in {time.time() - start_time:.2f} seconds")
        print(repr(self))

    @staticmethod
    def get_concept_set(data_object_dirs, output_folder = None):
        all_concept_ids = set()
        for dir_path in data_object_dirs:
            print(f"\nProcessing data object in {dir_path} ...")
            try:
                data_ref = Graph._process_data_object(dir_path)
                df = data_ref.collect()
                concept_ids = df["conceptId"].to_list()
                for cid in concept_ids:
                    try:
                        all_concept_ids.add(int(cid))
                    except Exception as ex:
                        print(f"Could not cast {cid} to int: {ex}")
            except Exception as e:
                print(f"An error occurred while processing {dir_path}: {e}")

        # Remove the 0 value if present.
        all_concept_ids.discard(0)
        # Add the root node concept id 441840 Clinical finding explicitly.
        all_concept_ids.add(441840)
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
        """
        data_object_dirs = [
            "/Users/xxx/data/plp/yyy",
            "/Users/xxx/data/plp/yyy",
            "/Users/xxx/data/plp/yyy"
        ]
        :param data_object_dirs:
        :param name:
        :return result_df:
        """


        self.ancestral_subgraph(self.get_concept_set(data_object_dirs), name=name)

    @staticmethod
    def _process_data_object(data_object_dir):
        zip_path = Path(data_object_dir) / "covariates"
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            sql_files = list(tmpdir.glob("*.sqlite")) + list(tmpdir.glob("*.duckdb"))
            if not sql_files:
                raise FileNotFoundError(
                    "No supported SQL database file (.sqlite or .duckdb) found in the zip archive.")

            data = str(sql_files[0])
            print(f"Found database file: {data}")

            if Path(data).suffix == ".sqlite":
                data = quote(data)
                data_ref = pl.read_database_uri(
                    "SELECT * FROM covariateRef", uri=f"sqlite://{data}"
                ).lazy()
            elif Path(data).suffix == ".duckdb":
                data = quote(data)
                # Create a local copy for DuckDB.
                destination = Path(data).parent.joinpath("python_copy.duckdb")
                path_copy = Path(shutil.copy(data, destination))
                conn = duckdb.connect(str(path_copy))
                data_ref = conn.sql("SELECT * FROM covariateRef").pl().lazy()
                conn.close()
                path_copy.unlink()  # Remove the temporary copy.
            else:
                raise ValueError("Only .sqlite and .duckdb files are supported")
            return data_ref

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
            if rustworkx_graph.in_degree(node_index) == 0 and rustworkx_graph.out_degree(node_index) > 0:
                root_nodes.append(rustworkx_graph[node_index])
        if len(root_nodes) == 0:
            raise ValueError("No root nodes found.")
        elif len(root_nodes) > 1:
            raise ValueError(f"Multiple root nodes found: {', '.join(map(str, root_nodes))}")

        print(f"Root node: {root_nodes[0]}")
        return root_nodes[0]

    def save(self, file_path):
        data = {
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