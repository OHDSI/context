"""Plot the original mean-rank and loss panels separately for each geometry."""
import argparse

import matplotlib.pyplot as plt
import polars as pl


def plot_geometry(data, geometry):
    dimensions = sorted(data["embedding_dim"].unique().to_list())
    negatives = sorted(data["negative_samples"].unique().to_list())
    fig, axes = plt.subplots(len(negatives), 2 * len(dimensions), squeeze=False, sharex=True,
                             sharey=True, figsize=(5 * len(dimensions), 3 * len(negatives)))
    for row, count in enumerate(negatives):
        for panel, directed in enumerate((True, False)):
            for col, dim in enumerate(dimensions):
                ax = axes[row, panel * len(dimensions) + col]
                loss_ax = ax.twinx()
                subset = data.filter((pl.col("directed") == directed) & (pl.col("embedding_dim") == dim) &
                                     (pl.col("negative_samples") == count))
                for group in subset.partition_by("experiment_id"):
                    group = group.sort("epoch")
                    label = f"Burn-in {group['burn_in_epochs'][0]}"
                    line, = ax.plot(group["epoch"], group["mean_rank"], marker="o", label=label)
                    loss_ax.plot(group["epoch"], group["loss"], ls="--", color=line.get_color(), label=label + " loss")
                ax.set_title(f"{'Directed' if directed else 'Undirected'}, {dim}D, {count} negatives")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Mean rank")
                loss_ax.set_ylabel("Loss")
                loss_ax.set_ylim(0, max(data["loss"].max(), 1e-12))
                if subset.height:
                    ax.legend(handles=ax.lines + loss_ax.lines, fontsize="small")
    fig.suptitle(geometry.capitalize())
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="eval_metrics.csv or eval_metrics_main.csv")
    args = parser.parse_args()
    data = pl.read_csv(args.metrics)
    if "geometry" not in data.columns:
        data = data.with_columns(pl.lit("hyperbolic").alias("geometry"))
    for (geometry,), group in data.partition_by(["geometry"], as_dict=True).items():
        plot_geometry(group, geometry)
    plt.show()


if __name__ == "__main__":
    main()
