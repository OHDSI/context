from dataclasses import dataclass

@dataclass
class Args:
    # output directory
    output_directory: str = '/Users/xxx/Desktop/output'
    # unique identifier for the experiment
    experiment_id: str = "exp_default"
    # graph identifier
    graph_id = "intermediate"
    # root node label
    root_node_label = 441840
    # treat the dataset during training and evaluation as directed graph
    directed: bool = False
    # concept dictionary
    concept_map = "/Users/xxx/data/vocabulary/snomed/CONCEPT.csv"
    # hierarchy file path
    hierarchy_file_path: str = "/Users/xxx/data/vocabulary/snomed/CONCEPT_ANCESTOR.csv"
    # concept identifier file
    concept_id_file: str = "/Users/xxx/Desktop/concept_ids.csv"
    # path to a model file to load a pre-trained model
    model_file: str = 'poincare_embeddings_snomed.pt'
    # number of models to save
    save_top: int = 5
    # number of epochs to train for
    epochs: int = 5000
    # learning rate
    learning_rate: float = 0.003*2048
    # perform burn-in
    burn_in: bool = True
    # learning rate divisor for burn-in
    burn_in_lr_divisor: float = 10
    # number of burn-in epochs
    burn_in_epochs: int = 20
    # batch size
    batch_size: int = 4096
    # embedding dimension
    embedding_dim: int = 3
    # curvature of the Poincare ball
    curvature: float = 1.0
    # compile the model
    compile: bool = False
    # device to use
    device: str = 'cpu'
    # negative samples
    negative_samples: int = 50
    # optimizer to use; either 'adam' or 'sgd'
    optimizer: str = 'sgd'
    # patience for early stopping
    early_stop_patience: int = 100
    # patience for learning rate reduction
    lr_reduce_patience: int = 50
    # factor for learning rate reduction
    lr_reduce_factor: int = 0.1
    # epoch interval for evaluating embedding
    eval_interval: int = 500
