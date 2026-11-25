# OHDSI/context

Train concept embeddings from a clinical taxonomy in Euclidean space or a Poincaré ball. Both models use the same concept rows, graph edges, conditional negative sampler and training objective. The default geometry is hyperbolic.

## Train both geometries on one graph

From a checkout with Python 3.12 and uv, run the following commands with your vocabulary and concept seed files.

```sh
uv sync --frozen
uv run python -m context --geometry both --concept_id_file concept_ids.csv \
  --concept_map vocabulary/CONCEPT.csv --relationship_file_path vocabulary/CONCEPT_RELATIONSHIP.csv \
  --root_node_label 441840 --output_directory output --experiment_id taxonomy \
  --embedding_dim 16 --optimizer adam --learning_rate 0.01 --seed 7
```

The dimension and learning rate illustrate the interface. Select them for the intended graph and task. Use `--geometry euclidean` or `--geometry hyperbolic` for a single fit. `--device mps` uses Apple Silicon and `--device cuda` selects an available NVIDIA GPU. Run `python -m context --help` for burn-in, early stopping, reconstruction frequency and checkpoint options.

| Setting | Euclidean | Hyperbolic |
| --- | --- | --- |
| Coordinates | Unconstrained real vectors | Open ball of radius `1/sqrt(c)` |
| Distance | L2 norm | Poincaré geodesic with sectional curvature `-c` |
| Updates | PyTorch SGD or Adam | hypll Riemannian SGD or Adam |
| Output for experiment `taxonomy` | `output/taxonomy/euclidean/` | `output/taxonomy/` (original location) |

`both` builds or loads one graph, resolves labels once, and fits each geometry using the same shuffled positive edges and negative draws at corresponding epochs. Both anchor the designated root at zero. Euclidean vectors are learned directly. The curvature option applies to the Poincaré fit.

## Graph inputs

- **SNOMED Is a.** Supply the tab-delimited OMOP `CONCEPT` and `CONCEPT_RELATIONSHIP` files and a CSV with one `conceptId` column. Active, standard SNOMED endpoints and active child-to-parent `Is a` rows define the taxonomy. The graph stores parent-to-child edges and retains intermediate nodes and multiple paths from covered seeds to the chosen root. In the saved graph, `Graph.load(path).extraction` records coverage and exclusions. The default root is OMOP concept 441840, Clinical finding.
- **OMOP ancestry.** To use a closure export, supply `--hierarchy_file_path vocabulary/CONCEPT_ANCESTOR.csv`. The builder retains one-step links, including links with a longer alternative path. This table provides ancestry without a relationship type.
- **Existing graphs.** Use `--graph_file path/to/graph.pkl` and `--graph_id ancestral` (or `full`). An existing experiment-local `graph.pkl` also takes precedence over vocabulary files. Use a fresh experiment ID to rebuild a graph after changing the seeds, root or vocabulary. Only open trusted graph pickles.
- **PLP seeds.** In R, source `R/extractPlpConcepts.R` and call `writePlpConcepts(plpData, "concept_ids.csv", analysisIds = ...)`. Omit `analysisIds` to use all mapped concepts. The helper reads the covariate dictionary and supports a lazy table via dplyr. Python callers can use `Graph.get_concept_set(plp_directories, output_folder)` for zipped SQLite or DuckDB dictionaries. Both paths extract underlying `conceptId` values.

## Training and graph reconstruction

Each observed edge competes against its own sampled alternatives. For source `u`, observed target `v`, and alternatives `w`, the loss is `log(1 + sum(exp(d(u,v) - d(u,w))))`. Sampling is uniform with replacement among eligible nodes. A sorted index over graph neighbors excludes self and all known positive targets without rejection loops or a dense adjacency matrix. Directed mode queries the stored edge direction. Undirected mode trains both orientations. A source adjacent to every candidate has no contrast, so its positives contribute only to reconstruction. The run summary counts these positives. Both distances are symmetric.

At each evaluation interval the package ranks the other graph nodes for each query with neighbors. Mean rank averages over positive edges and filters other known positives. Tied negatives count one half. Mean average precision (MAP) weights queries equally, includes all their true neighbors, and groups tied distances. Evaluation is batched in the fitted geometry. These are training-graph reconstruction diagnostics. A held-out design is needed to assess generalization. The exported checkpoint minimizes observed main-phase training loss.

Default Poincaré runs keep the original graph, TSV and checkpoint directory layout. Corrected per-edge loss, curvature and MAP can change numeric results relative to older versions. Older training passed the curvature argument through hypll's softplus. An older setting of 1 therefore used an effective magnitude of about 1.31326. New runs use the requested effective magnitude directly.

## Read the outputs

```python
from pathlib import Path
import torch
from context.export import load_model

folder = Path("output/taxonomy/euclidean")
artifact = torch.load(folder / "embedding.pt", weights_only=True, map_location="cpu")
concept_ids, vectors = artifact["concept_ids"], artifact["embeddings"]
epoch = artifact["epoch"]
model = load_model(next((folder / "models").glob(f"epoch_{epoch}-*.pt")))
```

`embedding.pt` contains row-aligned concept IDs, coordinates, geometry, effective curvature (null for Euclidean), anchor row, dimension and selected epoch. `vec.tsv` and `labels.tsv` retain the projector format. Labels fall back to identifiers when a concept name is unavailable. `training.json` records graph SHA-256, configuration, sampler counts and completed training losses. `models/` retains the strongest checkpoints by training loss, `eval_models/` holds checkpoints from evaluation epochs, and `eval_metrics.csv` records their reconstruction scores. New checkpoints load with `weights_only=True`. If a fit is interrupted after a checkpoint, the partial files remain in its printed `.context-fit-*` directory. Completed artifacts are published after each geometry finishes.

The Python entry point remains `context.train.train(Args(...))` and returns evaluation records. The examples include a parameter grid over a shared graph and a plot of mean rank and loss for each geometry. From the source tree, run the software checks with `uv run --frozen python -m pytest` and `Rscript --vanilla tests/test_plp_concepts.R`.
