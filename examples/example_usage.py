from context.graph import Graph
import networkx as nx
import polars as pl

concept_ids = [1, 2, 3, 4]
hierarchy = pl.read_csv("~/Desktop/snomed_vocabulary/CONCEPT_ANCESTOR.csv")

g = Graph(concept_ids, hierarchy)

