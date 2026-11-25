from dataclasses import dataclass
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
    # Magnitude of the negative sectional curvature for the Poincaré model.
    curvature: float = 1.0
    compile: bool = False
    device: str = "cpu"
    negative_samples: int = 50
    optimizer: Literal["adam", "sgd"] = "sgd"
    early_stop_patience: int = 100
    lr_reduce_patience: int = 50
    lr_reduce_factor: float = 0.1
    eval_interval: int = 500
    # Additional fields follow the original constructor's positional arguments.
    graph_id: str = "ancestral"
    root_node_label: int | None = 441840
    graph_file: str | None = None
    concept_map: str | None = None
    relationship_file_path: str | None = None
    geometry: Literal["hyperbolic", "euclidean", "both"] = "hyperbolic"
    seed: int = 1
    # Controls the Euclidean optimizer; root_node_label selects the taxonomy root.
    euclidean_root: Literal["fixed", "trainable"] = "fixed"
