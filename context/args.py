from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Args:
    output_directory: str = "output"
    experiment_id: str = "exp_default"
    directed: bool = False
    hierarchy_file_path: str = "CONCEPT_ANCESTOR.csv"
    concept_id_file: str = "concept_ids.csv"
    model_file: str = "poincare_embeddings_snomed.pt"
    save_top: int = 5
    epochs: int = 5000
    learning_rate: float = 0.003 * 2048
    burn_in: bool = True
    burn_in_lr_divisor: float = 10
    burn_in_epochs: int = 20
    batch_size: int = 4096
    embedding_dim: int = 3
    curvature: float = field(
        default=1.0, metadata={"help": "Magnitude of the negative sectional curvature for the Poincaré model."}
    )
    compile: bool = False
    device: str = "cpu"
    negative_samples: int = 50
    optimizer: Literal["adam", "sgd"] = "sgd"
    early_stop_patience: int = 100
    lr_reduce_patience: int = 50
    lr_reduce_factor: float = 0.1
    eval_interval: int = 500
    graph_id: str = field(default="ancestral", metadata={"help": "Subgraph name, or 'full' for the entire graph."})
    root_node_label: int | None = 441840
    graph_file: str | None = None
    concept_map: str | None = None
    relationship_file_path: str | None = None
    geometry: Literal["hyperbolic", "euclidean", "both"] = "hyperbolic"
    seed: int = 1
    euclidean_root: Literal["fixed", "trainable"] = field(
        default="fixed", metadata={"help": "Training policy for the Euclidean root selected by --root_node_label."}
    )
    taxonomy_scope: Literal["seeds", "subtree", "all"] = field(
        default="seeds", metadata={"help": "Seed-to-root paths, all root descendants, or the entire eligible SNOMED graph."}
    )
    snomed_concepts: Literal["standard", "active"] = field(
        default="standard", metadata={"help": "Use standard concepts or include all active SNOMED concepts, including structural nodes."}
    )
    eval_queries: int = field(default=256, metadata={"help": "Maximum fixed evaluation queries; zero ranks every eligible query."})
    query_seed: int = 2
    validation_fraction: float = field(default=0.0, metadata={"help": "Fraction each for validation and test edges; zero trains all edges."})
    split_seed: int = 1
    split_policy: Literal["random", "preserve_degrees"] = "random"
    retrieval_direction: Literal["observed", "parents", "children"] = "observed"
    resume: bool = field(default=False, metadata={"help": "Continue matching saved epoch state or reuse a completed fit."})
    checkpoint_interval: int = field(default=10, metadata={"help": "Main or burn-in epochs between durable recovery points, including epoch one."})
    sgd_updates: Literal["auto", "dense", "sparse"] = field(default="auto", metadata={"help": "Update touched rows for large sparse SGD batches on CPU or CUDA."})
    refit_from: str | None = field(default=None, metadata={"help": "Refit all edges using the selected run in this directory."})
    euclidean_distance_scale: float = field(default=1.0, metadata={"help": "Positive L2 multiplier in the Euclidean objective."})
    write_tsv: bool = field(default=True, metadata={"help": "Write projector TSV files alongside the ID-keyed tensor export."})
    threads: int | None = field(default=None, metadata={"help": "CPU threads used by PyTorch during fitting."})
    keep_eval_models: bool = field(default=True, metadata={"help": "Retain checkpoints from every scored epoch in addition to the selected model."})
    training_method: Literal["sampled", "landmark", "random", "adjacency_svd", "laplacian", "fastrp", "ancestor", "hope", "deepwalk", "node2vec"] = "sampled"
    landmarks: int | None = field(default=None, metadata={"help": "Graph landmark budget. Default is max(64, 2 × embedding dimension + 1), capped at the number of concepts."})
    landmark_edge_length: float | None = field(default=None, metadata={"help": "Target distance per training-graph hop. By default use one for Euclidean and an automatically bounded length for hyperbolic space."})
    diffusion_steps: int = 3
    degree_exponent: float = -0.5
    ancestor_decay: float = 0.5
    ancestor_mix: float = 0.5
    katz_beta: float = 0.1
    svd_iterations: int = 3
    spectral_tolerance: float = 1e-5
    spectral_iterations: int = 300
    walks_per_node: int = 4
    walk_length: int = 24
    walk_window: int = 5
    walk_epochs: int = 5
    walk_negatives: int = 5
    return_p: float = 1.0
    inout_q: float = 0.5
