from pathlib import Path

import polars as pl
import simple_parsing

from context.args import Args
from context.graph import Graph
from context.train import train

args = simple_parsing.parse(Args)
experiment_folder = Path(args.output_directory) / args.experiment_id
graph_file = experiment_folder / "graph.pkl"
experiment_folder.mkdir(parents=True, exist_ok=True)

if not graph_file.exists():
    df = pl.read_csv("/Users/xxx/Desktop/opehr_concepts.csv")
    melted_df = df.select(["ancestor_concept_id", "descendant_concept_id"]).melt().get_column("value")
    unique_concepts = melted_df.unique().to_list()
    concept_ids = unique_concepts

    hierarchy_path = "/Users/xxx/data/vocabulary/snomed/CONCEPT_ANCESTOR.csv"

    hierarchy = pl.read_csv(hierarchy_path, separator="\t")
    hierarchy = hierarchy.with_columns(pl.lit("subsumes").alias("edge_data"))
    filtered_hierarchy = hierarchy.filter(
        (pl.col("min_levels_of_separation") == 1) &
        (pl.col("max_levels_of_separation") == 1)
    ).select([
        pl.col('ancestor_concept_id').alias('source'),
        pl.col('descendant_concept_id').alias('target'),
        'edge_data'
    ])

    g = Graph(filtered_hierarchy)
    g.intermediate_subgraph(concept_ids)
    g.save(graph_file)

train(args=args)
