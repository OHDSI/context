# OHDSI/context

Train concept embeddings from a clinical taxonomy in Euclidean space or a Poincaré ball. Select one of twelve methods with `--training_method`. Sampled neighbor ranking and landmark fitting each support both geometries, giving fourteen method and geometry combinations in total. The default is sampled neighbor ranking in the Poincaré ball.

For ten methods, `context` implements the algorithm using NumPy, SciPy, rustworkx or PyTorch, with HyPLL for sampled Poincaré training. For DeepWalk and node2vec, `context` generates the graph walks and Gensim’s `Word2Vec` trains the embeddings. Install the optional `walks` extra with `uv sync --frozen --extra walks` to enable these two methods.

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

With the default sampled trainer, `both` builds or loads one graph, resolves labels once, and fits each geometry using the same shuffled positive edges and negative draws at corresponding epochs. By default both anchor the designated root at zero. Euclidean vectors are learned directly. The curvature option applies to the Poincaré fit.

### Euclidean root placement

The graph root selects which SNOMED descendants belong in the taxonomy. Use `--euclidean_root trainable` to fit its Euclidean vector along with the other concept vectors. Both policies start the root at zero. On PyTorch 2.5.1 the Apple MPS backend needs an explicit gradient mask to keep a fixed root there. The Poincaré fit in `--geometry both` keeps its named root anchored. Trainable-root Euclidean outputs go in `euclidean/trainable-root/`, so both Euclidean fits can share an experiment and graph.

For a Euclidean artifact with a named root, `vectors - vectors[artifact["graph_root_index"]]` places it at the origin after training and preserves all L2 distances up to rounding. `graph_root_index` identifies the named node. The existing `root_index` field identifies a frozen row and is null for a trainable root. Apply a chosen coordinate convention consistently when training and using a downstream model that consumes raw vectors. Parent–child direction is available from the graph.

## Fit from landmark distances

Use this option when you want the coordinates to reflect how many taxonomy steps separate concepts. Each landmark measures shortest-path distance to every concept, including paths through any of a concept’s parents. Euclidean landmark multidimensional scaling and hyperbolic L-hydra project those measurements into vectors. They use spectral factorization and require no negative samples or training epochs. [Euclidean landmark MDS](https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf) · [Hyperbolic L-hydra](https://arxiv.org/abs/2207.06775)

```sh
uv run --frozen python -m context --graph_file graph.pkl --graph_id ancestral \
  --root_node_label 441840 --training_method landmark --geometry both \
  --embedding_dim 16 --landmarks 64 --device cpu --seed 41 \
  --output_directory output --experiment_id taxonomy --eval_interval 0
```

The Poincaré table goes to `output/taxonomy/landmark/embedding.pt` and the Euclidean table to `output/taxonomy/landmark/euclidean/embedding.pt`. Use `--training_method sampled` for the original neighbor-ranking objective. The two trainers can share the graph and experiment directory. Epochs, negatives, optimizer and burn-in apply to sampled training. For landmark fitting, a positive `eval_interval` requests one retrieval evaluation. Both geometries reuse the same CPU shortest paths when requested together. Completed geometries can be reused with `--resume true`. An interrupted distance calculation starts again, and a fully staged export can finish publication without recomputing.

- **Graph and direction.** Fitting uses undirected hop distance on training links. It captures proximity. Parent and child retrieval can still be scored separately on held-out directed links. For a disconnected forest, a temporary hub connects one representative per weak component, with preference for its named root or an in-degree-zero node. The hub adds two hops between component representatives and preserves all original within-component distances. Between-component distances follow this convention. The hub has no exported row.
- **Origins and units.** The Poincaré fit moves the named root to the origin by an isometry. Euclidean fitting uses translation with `--euclidean_root fixed` and landmark-centered coordinates with `--euclidean_root trainable`. The default target is one unit per hop in Euclidean space. For Poincaré the default is the smaller of one and `6 / (sqrt(curvature) × maximum landmark hops)`, which limits the numerical range of the hyperbolic cosine. Override with a positive `--landmark_edge_length`. The actual length and effective curvature are saved.
- **Resolution.** The first landmark is the named root when supplied. Farthest-point selection covers the graph, with ties resolved by the seed. The default budget is the smaller of the concept count and `max(64, 2 × embedding_dim + 1)`. More landmarks can distinguish additional branches. Identical landmark-distance profiles produce identical coordinates. The receipt records unique profiles, distinct exported vectors, the number of concepts sharing a vector, the largest such group and the effective spectral dimension. Rank-deficient graphs are zero-padded to retain the requested export width. Eigenvector orientation can vary with the numerical libraries.
- **Scale and evaluation.** With *N* nodes and *L* landmarks the hop table holds *N × L* 32-bit integers. Only the *L × L* landmark matrix is factorized. Use `--write_tsv false` for tensor-only transfer. A validation split is applied before forming any landmark distances or links across components. Reciprocal edges are assigned to the same fold. Use the existing validation/refit workflow below to select the geometry and budget and then refit on all edges.

## Euclidean methods and assignment controls

Select `--geometry euclidean --device cpu` and one of the following values for `--training_method`. All methods use the same graph input, fixed or free Euclidean origin, content-checked export and graph-only holdout/refit interface. Vectors for each method are saved in `output/<experiment>/<method>/euclidean/embedding.pt`. The graph stores parent-to-child links. HOPE, ancestor pooling and topological order use that orientation.

| Method | Coordinates and controls | Limitation to inspect |
| --- | --- | --- |
| `random` | Seeded Gaussian vectors for each concept ID. For a fixed concept set these are independent of the edges. | A reference for what the downstream architecture can learn from code identity alone. |
| `adjacency_svd` | Left singular vectors of the undirected adjacency, scaled by square roots of the singular values. | Nodes with identical neighborhoods can coincide. Common hubs influence the leading axes. |
| `laplacian` | Laplacian Eigenmaps, solving the degree-weighted graph eigenproblem. Nonconstant modes encourage smooth variation over direct links. | Isolates have zero native coordinates. Between-component placement is unconstrained. Small spectral gaps can slow convergence. Inspect the recorded eigen residual and tune `--spectral_iterations` and `--spectral_tolerance`. |
| `fastrp` | FastRP with a sparse ternary projection (density 1/3), degree weighting and equal weights for walk powers 1 through `--diffusion_steps`. Tune `--degree_exponent`. | Longer walks smooth local differences. Sparse projections can yield ties in small dimensions. |
| `ancestor` | Combines a Gaussian code vector with the weighted average of its parents and more distant ancestors. Every parent contributes. `--ancestor_mix` goes from 0 (identity) to 1 (pure ancestor signal). `--ancestor_decay` and `--diffusion_steps` control reach. | Multiple paths affect the weights. Siblings with the same parents can coincide under pure pooling. Roots retain their identity vector. |
| `hope` | HOPE with directed Katz proximity, summing parent-to-descendant path counts weighted by `--katz_beta` to the power of path length. Sparse triangular solves on a DAG and low-rank SVD avoid constructing the full closure. | This implementation requires a DAG. The contribution of alternate paths depends on beta. |
| `order` | Each column holds the rank in a randomized topological ordering. Parents precede descendants in every coordinate. | Unrelated families can appear comparable at finite width. Rank gaps do not encode graph distance. |
| `graph_autoencoder` | Featureless two-layer GCN on the undirected training graph with normalized self-loops and an inner-product decoder. An implicit identity matrix provides node features. | Each node has a learned weight. The symmetric decoder gives the same score to both edge directions. Training cost depends on hidden width and epoch budget. |
| `deepwalk` | Uniform undirected walks and Skip-gram with hierarchical softmax. The softmax tree models token frequencies. | The walk context can be dominated by common nodes. |
| `node2vec` | Second-order walks with return parameter `--return_p` and outward parameter `--inout_q`, and Skip-gram with negative sampling. | `p`, `q`, walk length and window change the neighborhoods being learned. |

For HOPE, each endpoint role receives half the requested width, rounded down. For odd widths the trailing column is zero. To recover a native directed score, undo the Euclidean translation and scaling with `native = exported * distance_scale + native_origin`. Take the dot product of the source half for a parent with the target half for a child. The standard package retrieval metric remains Euclidean distance so all exported point tables can be compared on a common score. For both SVD methods the solver uses exact factorization for graphs of at most 256 nodes and randomized block SVD above that threshold. Tune `--svd_iterations` for approximation quality. Laplacian Eigenmaps uses a sparse symmetric eigensolver above the small-graph threshold.

```sh
uv sync --frozen --extra walks
uv run --frozen --extra walks python -m context --graph_file graph.pkl --geometry euclidean \
  --training_method node2vec --root_node_label 441840 --walks_per_node 4 --walk_length 24 \
  --walk_window 5 --walk_epochs 5 --return_p 1 --inout_q 0.5 --threads 1 --experiment_id taxonomy
```

The order baseline uses independently randomized vertex priorities to create linear extensions of the DAG. Its fitted table preserves the direction of all training edges coordinate by coordinate. The graph autoencoder uses PyTorch sparse message passing and balanced sampled nonedge reconstruction. The positive-edge budget is per optimizer update, and negative pairs are uniformly sampled from off-diagonal unobserved pairs in the training graph. Tune `--gae_hidden`, `--gae_epochs`, `--gae_lr`, `--gae_edge_budget` and `--gae_negatives`. An epoch checkpoint preserves the neural weights, Adam state and pair-sampling RNG. [Order dimension](https://fa.ewi.tudelft.nl/~hart/set_theory/material/dushnik_miller-partially_ordered_sets.pdf) · [Graph autoencoders](https://arxiv.org/abs/1611.07308).

DeepWalk and node2vec use the optional Gensim dependency, a streamed seeded walk corpus and an epoch-wise linear learning-rate schedule. `--walk_negatives` applies to node2vec. `--threads 1` gives reproducible single-worker training in the pinned environment. Additional workers permit asynchronous updates. The walk trainer checkpoints after the first and last epochs and at `--checkpoint_interval`. `--resume true` continues the saved optimizer state in Gensim, including its context weights and RNG. The spectral, projection, order and pooling methods run once per fit and reuse verified complete exports on resume.

A shuffled-assignment control preserves the set of trained vectors, geometry and named root while assigning nonroot codes to other rows. The command writes a new export and a reproducible destination-to-source row map. The `degree` option shuffles within exact parent counts and log-binned child counts using only training edges. Set `--matching none` for an unrestricted shuffle. The output reports the number of codes whose actual vector changed, including the effect of ties.

```sh
uv run --frozen python -m context.controls --source output/taxonomy/node2vec/euclidean/embedding.pt \
  --destination output/taxonomy/shuffled-node2vec --matching degree --graph-file graph.pkl --seed 521
```

[Laplacian Eigenmaps](https://papers.nips.cc/paper_files/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html) · [Randomized SVD](https://arxiv.org/abs/0909.4061) · [FastRP](https://arxiv.org/abs/1908.11512) · [HOPE](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf) · [DeepWalk](https://arxiv.org/abs/1403.6652) · [node2vec](https://arxiv.org/abs/1607.00653).

## Check a supplied SNOMED vocabulary

The optional integration test reads the two OMOP vocabulary tables, checks every active concept and direct `Is a` edge against an independent SQL query, and verifies the concept IDs and coordinates exported by each method. Use a private temporary directory for the graph and weights. The short settings exercise package behavior.

```sh
CONTEXT_SNOMED_VOCABULARY=/path/to/authorized/vocabulary \
  uv run --frozen --extra walks pytest -q tests/test_snomed_vocabulary.py
```

## Graph inputs

- **SNOMED Is a.** Supply the tab-delimited OMOP `CONCEPT` and `CONCEPT_RELATIONSHIP` files and a CSV with one `conceptId` column. Active, standard SNOMED endpoints and active child-to-parent `Is a` rows define the taxonomy. The graph stores parent-to-child edges and retains intermediate nodes and multiple paths from covered seeds to the chosen root. In the saved graph, `Graph.load(path).extraction` records coverage and exclusions. The default root is OMOP concept 441840, Clinical finding.
- **OMOP ancestry.** To use a closure export, supply `--hierarchy_file_path vocabulary/CONCEPT_ANCESTOR.csv`. The builder retains one-step links, including links with a longer alternative path. This table provides ancestry without a relationship type.
- **Existing graphs.** Use `--graph_file path/to/graph.pkl` and `--graph_id ancestral` (or `full`). A newly built experiment-local cache records content hashes of graph inputs, root, selection and scope. Changes require a fresh experiment ID. To reuse an older graph without its source vocabulary, pass it explicitly through `--graph_file`. Only open trusted graph pickles.
- **PLP seeds.** In R, source `R/extractPlpConcepts.R` and call `writePlpConcepts(plpData, "concept_ids.csv", analysisIds = ...)`. Omit `analysisIds` to use all mapped concepts. The helper reads the covariate dictionary and supports a lazy table via dplyr. Python callers can use `Graph.get_concept_set(plp_directories, output_folder)` for zipped SQLite or DuckDB dictionaries or pass an extracted database file directly. Direct database input avoids decompressing a large covariate archive. Both paths extract underlying `conceptId` values.

## Training and graph reconstruction

Each observed edge competes against its own sampled alternatives. For source `u`, observed target `v`, and alternatives `w`, the loss is `log(1 + sum(exp(d(u,v) - d(u,w))))`. Sampling is uniform with replacement among eligible nodes. A sorted index over graph neighbors excludes self and all known positive targets without rejection loops or a dense adjacency matrix. Directed mode queries the stored edge direction. Undirected mode trains both orientations. A source adjacent to every candidate has no contrast, so its positives contribute only to reconstruction. The run summary counts these positives. Both distances are symmetric.

At each evaluation interval the trainer ranks all other graph nodes for up to 256 uniformly selected eligible queries. The same query seed fixes membership across epochs and paired geometries. Set `--eval_queries 0` for the exhaustive historical reconstruction statistic. The standalone evaluation functions default to all eligible queries. Mean rank averages over positive edges and filters other known positives. Tied negatives count one half. Mean average precision (MAP) weights queries equally, includes all their true neighbors, and groups tied distances. Evaluation is batched in the fitted geometry. These are training-graph reconstruction diagnostics. A held-out design is needed to assess generalization. The default full-graph fit exports the epoch with minimum observed main-phase training loss. The optional held-out workflow below uses validation MAP.

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

`embedding.pt` contains row-aligned concept IDs, coordinates, geometry, effective curvature (null for Euclidean), anchor row, dimension and selected epoch. `vec.tsv` and `labels.tsv` retain the projector format. Labels fall back to identifiers when a concept name is unavailable. `training.json` records graph SHA-256, configuration, sampler counts and completed training losses. `models/` retains the strongest checkpoints by training loss, `eval_models/` holds checkpoints from evaluation epochs, and `eval_metrics.csv` records their reconstruction scores. New checkpoints load with `weights_only=True`. Each geometry publishes its completed artifacts separately. The recovery procedure below continues an interrupted fit.

The Python entry point remains `context.train.train(Args(...))` and returns evaluation records. The examples include a parameter grid over a shared graph and a plot of mean rank and loss for each geometry. From the source tree, run the software checks with `uv run --frozen python -m pytest` and `Rscript --vanilla tests/test_plp_concepts.R`.

## Working with large taxonomies

Choose the graph population explicitly. With typed OMOP inputs, `--taxonomy_scope seeds` retains every path from a covered concept to the nominated root, including multiple parents. `--taxonomy_scope subtree` includes all descendants of that root. `--taxonomy_scope all` includes every active standard SNOMED concept in the supplied release, including disconnected branches and isolates. Use `--snomed_concepts active` to include active nonstandard structural concepts and their links as well. These two choices are independent. Removing an intermediate structural node can disconnect otherwise related standard concepts. The latter two graph scopes need no concept seed file. For the study the named anchor is Clinical finding (OMOP 441840). In a complete vocabulary graph that node can anchor the coordinates even though other branches have different roots. Isolates have no observed positive edge. Their vectors can receive negative updates and have no positive hierarchy evidence.

Graph construction uses compressed sparse rows and native directed traversal. Node labels, their row order, edges and edge payloads round-trip in a versioned compact cache. The reader also accepts earlier context graph pickles. `Graph.selective_subgraph(concept_ids)` supports a small panel of retained concepts. An edge labelled `ancestor_path` connects two retained concepts through omitted intermediates. For direct-parent evaluation use the original typed edges.

For an initial timing pass on your hardware, set `--epochs 1 --burn_in false`. Record the graph size, direction, batch size, negatives, dimension, optimizer, device and evaluation scope when comparing speeds. Undirected training normally doubles the positive orientations. The cost of dense Adam and its moment arrays grows with the full concept table for each update. On CPU and CUDA the automatic SGD policy uses row updates for Poincaré batches that touch a small fraction of that table. Both routes use hypll's exponential map. Set `--sgd_updates dense` or `sparse` to choose explicitly for SGD. MPS uses dense updates. `--threads` sets PyTorch's CPU threads for the fitting call. Use separate processes for concurrent jobs with different thread settings.

Training holds a sparse graph index, one embedding row per node, optimizer state, and the current minibatch. Reconstruction ranks against every candidate in blocks. With Q queries, N nodes and dimension D, distance computation scales as Q × N × D. Exhaustive reconstruction can therefore dominate the run. Distances are ranked on CPU in float64 using squared chord length and, for Poincaré, its candidate-specific ball margin. The omitted factors and transformations are constant or strictly increasing for a fixed query. The implementation counts ties exactly. Choose `--eval_queries` for the precision needed and `--eval_interval` for how often to pay that cost. `--write_tsv false --keep_eval_models false` saves disk space while retaining the tensor, its checksum, the top checkpoints and the selected epoch.

## Held-out parent retrieval and full-graph refit

The following example assumes a trusted saved parent-to-child graph. Each geometry uses the same graph rows and edge partitions. For each split, training negatives exclude self and the observed training neighbors. Validation filters previously known training positives. Test retrieval also filters validation edges. All graph nodes remain candidates. Reciprocal edges stay together for undirected training.

```sh
uv run --frozen python -m context --graph_file graph.pkl --geometry both \
  --directed true --validation_fraction 0.1 --split_policy preserve_degrees \
  --retrieval_direction parents --split_seed 11 --query_seed 17 --eval_queries 256 \
  --epochs 100 --eval_interval 5 --optimizer adam --learning_rate 0.01 \
  --embedding_dim 16 --keep_eval_models false --output_directory results --experiment_id selection
```

Here 10% of the independent edge groups go to validation and 10% to test. The `random` policy also permits nodes without training positives. `preserve_degrees` protects an incoming and an outgoing edge wherever those degrees are positive and reports an error if the requested fraction exceeds the remaining capacity. In a rooted DAG this preserves a training path to each node. The score for selection is query-weighted validation MAP. Use the same query limit and seeds when comparing geometries. The selected checkpoint is evaluated once on the test edges per run. Use validation results alone for choosing between runs. The partitions, number of queries, candidate count, filtered mean rank, MAP, actual learning rates and selected epoch are recorded.

For example, after choosing the Euclidean run by validation MAP, refit its policy on every graph edge. Pass the Poincaré selection directory instead if it won. The refit uses the recorded main-phase learning rates through the selected epoch, replays its configured burn-in, and exports the final epoch.

```sh
uv run --frozen python -m context --refit_from results/selection/euclidean \
  --graph_file graph.pkl --experiment_id all-edges --output_directory results
```

## Interruption, transfer and downstream use

Rerun the same fitting command with `--resume true`. Keep the output directory including its hidden `.context-fit-*` staging folders. The trainer atomically saves the embedding, optimizer and scheduler state, stopping history, edge-order generator and negative-sampler generator after the first epoch of each phase and every ten epochs by default. `--checkpoint_interval 1` saves every completed epoch. Work since the last durable recovery point is replayed. Each completed geometry can be reused after artifact verification. Use the same package revision, device setting, PyTorch runtime and CPU thread count when continuing on another machine. Copy or explicitly point to the same graph and label source. CPU replay is tested for exact agreement in the pinned environment. Accelerator accumulation can differ at floating-point precision.

`context.export.load_embedding(path)` checks the sidecar SHA-256, row alignment, geometry and finite coordinates. The two top-level tensors use OHDSI DeepPatientLevelPrediction's mapped concept IDs and float32 vectors. Use `CustomEmbeddings` for native Euclidean coordinates and `PoincareEmbeddings` with the exported effective curvature for ball coordinates. Each binary covariate is matched by concept ID. The PLP dictionary and the model's covariate columns can have a different order. For older two-tensor exports without metadata supply `geometry=` and, for Poincaré, the known effective `curvature=` to the reader. The optional `--euclidean_distance_scale` reproduces objectives using a positive multiplier on L2 distance. Coordinates retain their native units.

Old research runs and study-specific clinical estimators retain their pinned implementations for exact historical replay. The training loss, candidate grouping and validation choices in this branch are recorded with its new fits.
