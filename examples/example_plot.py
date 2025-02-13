import polars as pl
import matplotlib.pyplot as plt

# Load the CSV file
df = pl.read_csv("/Users/xxx/Desktop/eval.csv")

# Split into directed and undirected experiments.
df_directed   = df.filter(pl.col("directed") == "true")
df_undirected = df.filter(pl.col("directed") == "false")

# Compute a global maximum loss based on the entire dataset
global_max_loss = df.select(pl.col("loss").max()).item()

# For determining grid layout we assume the hyperparameters (embedding_dim and negative_samples)
# are identical (or at least consistent) between directed and undirected experiments.
embedding_dims = sorted(df["embedding_dim"].unique().to_list())
neighbour_counts = sorted(df["negative_samples"].unique().to_list())

# We'll have two panels: one for directed and one for undirected.
# Each panel is a grid with rows = len(neighbour_counts) and columns = len(embedding_dims).
# So the total figure will have nrows = len(neighbour_counts) and ncols = 2 * len(embedding_dims)
n_rows = len(neighbour_counts)
n_cols_per_panel = len(embedding_dims)
total_cols = 2 * n_cols_per_panel

fig, axes = plt.subplots(nrows=n_rows, ncols=total_cols, figsize=(24, 12), sharex=True, sharey=True)

# Get a default color cycle to use the same color for corresponding mean rank and loss.
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# --- Plot for Directed experiments ---
for row_idx, n in enumerate(neighbour_counts):
    for col_idx, ed in enumerate(embedding_dims):
        # For directed experiments, use the left panel (first n_cols_per_panel columns)
        ax = axes[row_idx, col_idx]
        ax2 = ax.twinx()

        # Filter the relevant subset
        subset = df_directed.filter((pl.col("embedding_dim") == ed) &
                                    (pl.col("negative_samples") == n))

        experiment_ids = subset["experiment_id"].unique().to_list()

        for idx, exp_id in enumerate(experiment_ids):
            group = subset.filter(pl.col("experiment_id") == exp_id).sort("epoch")
            epochs = group["epoch"].to_list()
            mean_ranks = group["mean_rank"].to_list()
            losses = group["loss"].to_list()
            # Annotate by burn-in epochs (assumed constant within group)
            burn_in_epochs = group["burn_in_epochs"].to_list()[0]
            color = colors[idx % len(colors)]
            ax.plot(epochs, mean_ranks, marker="o",
                    label=f"Burn-in {burn_in_epochs}", color=color)
            ax2.plot(epochs, losses, marker="s", linestyle="--",
                     label="Loss", color=color)

        ax.set_title(f"Dim: {ed}, Neighbours: {n}")
        ax.grid(True)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Epoch")
        if col_idx == 0:
            ax.set_ylabel("Mean Rank")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(0, global_max_loss)

        # Combine legends from both axes.
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize="small", loc="best")

# --- Plot for Undirected experiments ---
for row_idx, n in enumerate(neighbour_counts):
    for col_idx, ed in enumerate(embedding_dims):
        # For undirected experiments, use the right panel,
        # i.e. column index offset by n_cols_per_panel.
        ax = axes[row_idx, col_idx + n_cols_per_panel]
        ax2 = ax.twinx()

        subset = df_undirected.filter((pl.col("embedding_dim") == ed) &
                                      (pl.col("negative_samples") == n))
        experiment_ids = subset["experiment_id"].unique().to_list()

        for idx, exp_id in enumerate(experiment_ids):
            group = subset.filter(pl.col("experiment_id") == exp_id).sort("epoch")
            epochs = group["epoch"].to_list()
            mean_ranks = group["mean_rank"].to_list()
            losses = group["loss"].to_list()
            burn_in_epochs = group["burn_in_epochs"].to_list()[0]
            color = colors[idx % len(colors)]
            ax.plot(epochs, mean_ranks, marker="o",
                    label=f"Burn-in {burn_in_epochs}", color=color)
            ax2.plot(epochs, losses, marker="s", linestyle="--",
                     label="Loss", color=color)

        ax.set_title(f"Dim: {ed}, Neighbours: {n}")
        ax.grid(True)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Epoch")
        if col_idx == 0:
            ax.set_ylabel("Mean Rank")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(0, global_max_loss)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize="small", loc="best")

# Add super-title annotations for the two panels
# We'll position these using figure text in normalized coordinates.
fig.text(0.25, 0.97, "Directed", ha="center", va="center", fontsize=16)
fig.text(0.75, 0.97, "Undirected", ha="center", va="center", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.94])  # reserve some space at top for the panel titles
plt.show()