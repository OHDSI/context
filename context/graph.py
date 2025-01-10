import networkx as nx
import time
import polars as pl


class Graph:
    def __init__(self, hierarchy):
        """
        :param concept_ids: list of concept ids to create a hierarchy from
        :param hierarchy: full hierarchy edge list (e.g. SNOMED's
        CONCEPT_ANCESTOR.csv)
        """
        self.full_graph = nx.DiGraph()
        self.subgraph = {}

        start = time.time()
        filtered_hierarchy = hierarchy.filter(
            (pl.col("min_levels_of_separation") == 1) &
            (pl.col("max_levels_of_separation") == 1)
        ).select(['ancestor_concept_id', 'descendant_concept_id']
        ).to_numpy().tolist()
        self.full_graph = nx.DiGraph(filtered_hierarchy)
        delta = time.time() - start

        print(f"Processed edge list in {delta:.2f} seconds")
        print(repr(self))

    def __repr__(self):
        subgraph_info = ", ".join(
            f"{name}: ({subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges)"
            for name, subgraph in self.subgraph.items()
        )
        return (
            f"Directed graph has {self.full_graph.number_of_nodes()} nodes "
            f"and {self.full_graph.number_of_edges()} edges; "
            f"Subgraphs: [{subgraph_info}]")

    def intermediate_subgraph(self, concept_ids, name="intermediate"):
        start = time.time()
        self.subgraph[name] = self.full_graph.subgraph(concept_ids).copy()
        delta = time.time() - start
        print(f"Created {name} subgraph in {delta:.2f} seconds")
