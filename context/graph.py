import networkx as nx
import time


class Graph:
    def __init__(self, concept_ids, hierarchy):
        """
        :param concept_ids: list of concept ids to create a hierarchy from
        :param hierarchy: full hierarchy edge list (e.g. SNOMED's
        CONCEPT_ANCESTOR.csv)
        """
        self.graph = nx.DiGraph()
        start = time.time()



        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")

    def add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def remove_edge(self, u, v):
        self.graph.remove_edge(u, v)

    def has_edge(self, u, v):
        return self.graph.has_edge(u, v)

    def get_neighbors(self, u):
        return list(self.graph.successors(u))
