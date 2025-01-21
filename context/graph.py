import rustworkx as rx
import networkx as nx
import time
import polars as pl
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, hierarchy):
        """
        :param hierarchy: full hierarchy edge list (e.g. SNOMED's
        CONCEPT_ANCESTOR.csv)
        """
        self.full_graph = rx.PyDiGraph()
        self.subgraph = {}
        self.node_map = {}
        self.node_label = {}

        start = time.time()

        for source, target, edge_data in hierarchy.iter_rows(named=False):
            if source not in self.node_map:
                self.node_map[source] = self.full_graph.add_node(source)
            if target not in self.node_map:
                self.node_map[target] = self.full_graph.add_node(target)
            self.full_graph.add_edge(self.node_map[source], self.node_map[target], edge_data)

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

    def __to_networkx(self, rustworkx_graph):
        start = time.time()

        nx_graph = nx.DiGraph()
        for node_index in rustworkx_graph.node_indexes():
            nx_graph.add_node(rustworkx_graph[node_index])
        for source, target, edge_data in rustworkx_graph.weighted_edge_list():
            nx_graph.add_edge(rustworkx_graph[source], rustworkx_graph[target], data=edge_data)

        delta = time.time() - start
        print(f"Converted to networkx graph in {delta:.2f} seconds")
        return nx_graph

    def plot(self, graph_name="full", edge_labels=True):
        if graph_name == "full":
            rustworkx_graph = self.full_graph
        elif graph_name in self.subgraph:
            rustworkx_graph = self.subgraph[graph_name]
        else:
            raise ValueError(f"Graph '{graph_name}' not found. Available options are 'full' and subgraphs: {list(self.subgraph.keys())}")

        nx_graph = self.__to_networkx(rustworkx_graph)
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_size=250, node_color="lightblue", font_size=10)
        if edge_labels:
            edge_data = nx.get_edge_attributes(nx_graph, 'data')
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_data)
        plt.show()

    def selective_subgraph(self, concept_ids, name="selective"):
        # todo: a subgraph that only contains the nodes specified in concept_ids and must create new edges between them.
        #  It may be best to use intermediate subgraph as basis.
        return None

    def intermediate_subgraph(self, concept_ids, name = "intermediate"):
        """
        :param concept_ids: list of concept IDs for which to create a subgraph
        :param name: custom name of the subgraph
        """
        start_time = time.time()

        subgraph = rx.PyDiGraph()
        seen_nodes = set()
        subgraph_node_map = {}

        for u in concept_ids:
            u_index = self.node_map[u]
            for v in concept_ids:
                if u != v:
                    v_index = self.node_map[v]
                    all_paths = list(rx.all_simple_paths(self.full_graph, u_index, v_index))
                    for path in all_paths:
                        seen_nodes.update(path)

        for node_index in seen_nodes:
            node_label = self.full_graph[node_index]
            if node_label not in subgraph_node_map:
                subgraph_node_map[node_label] = subgraph.add_node(node_label)

        for u, v, edge_data in self.full_graph.weighted_edge_list():
            if u in seen_nodes and v in seen_nodes:
                subgraph.add_edge(subgraph_node_map[self.full_graph[u]], subgraph_node_map[self.full_graph[v]], edge_data)

        self.subgraph[name] = subgraph
        print(f"Created {name} subgraph in {time.time() - start_time:.2f} seconds")

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
        print(f"Nodes in {graph_name} graph: [{', '.join(node_list)}]")
