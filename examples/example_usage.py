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
    # # use plp data to generate args.concept_id_file holding the concept set used in this example
    # data_object_dirs = [
    #     "/Users/xxx/data/plp/yyy",
    #     "/Users/xxx/data/plp/yyy",
    #     "/Users/xxx/data/plp/yyy"
    # ]
    # concept_ids = Graph.get_concept_set(data_object_dirs, args.concept_id_file)
    # # or directly generate subgraph from plp data
    # Graph.ancestral_subgraph(data_object_dirs)

    df = pl.read_csv(args.concept_id_file)
    concept_ids = df["conceptId"].unique().to_list()


    hierarchy = pl.read_csv(args.hierarchy_file_path, separator="\t")
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
    g.ancestral_subgraph(concept_ids)
    g.save(graph_file)

train(args=args)
