from context.graph import Graph
import networkx as nx
import polars as pl

concept_ids = [440921,35625043,442564,46270406,604180,4051479,441840,4185197,
               313800,606421,606771,4146460,77630,4042834,193666,4344395,
               444131,619020,432795,194702,443957,40482935,443407,605283,
               200219,4266830,192763,4319280,197311,195562,607784,433385]
hierarchy_path = "~/data/vocabulary/snomed/CONCEPT_ANCESTOR.csv"
hierarchy = pl.read_csv(hierarchy_path, separator="\t")
print(hierarchy[1:5])
g = Graph(hierarchy)

g.intermediate_subgraph(concept_ids)

print(g)