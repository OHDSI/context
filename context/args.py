from dataclasses import dataclass

@dataclass
class Args:
    # path to a save and load a graph file
    graph_file: str = '/Users/xxx/Desktop/output/graph.pkl'
    # graph identifier
    graph_id = "intermediate"
    # root node label
    root_node_label = 441840
    # path to a model file to load a pre-trained model
    model_file: str = 'poincare_embeddings_snomed.pt'
    # output directory
    output_directory: str = '/Users/xxx/Desktop/output/models'
    # number of models to save
    save_top: int = 5
    # number of epochs to train for
    epochs: int = 1
    # learning rate
    learning_rate: float = 0.000000001# 0.003*16
    # perform burn-in
    burn_in: bool = True
    # learning rate divisor for burn-in
    burn_in_lr_divisor: float = 10
    # number of burn-in epochs
    burn_in_epochs: int = 1
    # batch size
    batch_size: int = 4*32
    # embedding dimension
    embedding_dim: int = 3
    # curvature of the Poincare ball
    curvature: float = 1.0
    # compile the model
    compile: bool = False
    # device to use
    device: str = 'mps'
    # negative samples
    negative_samples: int = 10
    # optimizer to use; either 'adam' or 'sgd'
    optimizer: str = 'sgd'
    # patience for early stopping
    early_stop_patience: int = 1000
