import copy
import itertools
from pathlib import Path
import polars as pl
import simple_parsing
import shutil

from context.args import Args
from context.graph import Graph
from context.train import train

def create_common_graph(common_graph_file: Path, args: Args):
    if common_graph_file.exists():
        return
    df = pl.read_csv(args.concept_id_file)
    unique_concepts = df["conceptId"].unique().to_list()
    hierarchy = pl.read_csv(args.hierarchy_file_path, separator="\t")
    hierarchy = hierarchy.with_columns(pl.lit("subsumes").alias("edge_data"))
    filtered_hierarchy = hierarchy.filter((pl.col("min_levels_of_separation") == 1) & (pl.col("max_levels_of_separation") == 1))\
                                .select([pl.col("ancestor_concept_id").alias("source"),
                                         pl.col("descendant_concept_id").alias("target"),
                                         "edge_data"])
    g = Graph(filtered_hierarchy)
    g.ancestral_subgraph(unique_concepts)
    g.save(common_graph_file)

def main():
    args = simple_parsing.parse(Args)
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    common_graph_file = output_dir / "graph.pkl"
    create_common_graph(common_graph_file, args)
    param_grid = {
        "directed": [True, False],
        "negative_samples": [10, 50, 100],
        "embedding_dim": [3, 10, 30, 100],
        "burn_in_epochs": [10, 100]
    }
    param_keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*(param_grid[key] for key in param_keys)))
    all_eval_records = []
    experiment_counter = 0
    main_csv_file = output_dir / "eval_metrics_main.csv"
    backup_csv_file = output_dir / "eval_metrics_main.bak.csv"
    for combination in all_combinations:
        exp_config = dict(zip(param_keys, combination))
        exp_args = copy.deepcopy(args)
        exp_args.directed = exp_config["directed"]
        exp_args.negative_samples = exp_config["negative_samples"]
        exp_args.embedding_dim = exp_config["embedding_dim"]
        exp_args.burn_in_epochs = exp_config["burn_in_epochs"]
        exp_args.experiment_id = f"exp_{experiment_counter:03d}"
        experiment_counter += 1
        experiment_folder = output_dir / exp_args.experiment_id
        if experiment_folder.exists() and (experiment_folder / "eval_metrics.csv").exists():
            eval_records = pl.read_csv(str(experiment_folder / "eval_metrics.csv")).to_dicts()
        else:
            experiment_folder.mkdir(parents=True, exist_ok=True)
            local_graph_file = experiment_folder / "graph.pkl"
            if not local_graph_file.exists():
                shutil.copy(common_graph_file, local_graph_file)
            eval_records = train(args=exp_args)
        all_eval_records.extend(eval_records)
        combined_df = pl.DataFrame(all_eval_records)
        if main_csv_file.exists():
            shutil.copy(main_csv_file, backup_csv_file)
        combined_df.write_csv(str(main_csv_file))

if __name__ == "__main__":
    main()