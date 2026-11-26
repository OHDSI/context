import math

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from context.export import atomic_save


def normalized_graph(adjacency):
    loops = adjacency.astype(np.float32) + sparse.eye(adjacency.shape[0], format="csr", dtype=np.float32)
    degree = np.asarray(loops.sum(axis=1)).ravel()
    inv = sparse.diags(1 / np.sqrt(degree))
    normalized = (inv @ loops @ inv).tocoo()
    indices = torch.from_numpy(np.stack((normalized.row, normalized.col)).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(normalized.data), normalized.shape).coalesce()


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, n, hidden, dimensions):
        super().__init__()
        self.node_weights = torch.nn.Parameter(torch.empty(n, hidden))
        self.projection = torch.nn.Parameter(torch.empty(hidden, dimensions))
        torch.nn.init.normal_(self.node_weights, std=1 / math.sqrt(hidden))
        torch.nn.init.xavier_uniform_(self.projection)

    def forward(self, graph):
        hidden = F.relu(torch.sparse.mm(graph, self.node_weights))
        return torch.sparse.mm(graph, hidden @ self.projection)


class ReconstructionPairs:
    def __init__(self, adjacency):
        self.n = adjacency.shape[0]
        upper = sparse.triu(adjacency, k=1, format="coo")
        self.positives = np.column_stack((upper.row, upper.col)).astype(np.int64)
        rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(adjacency.indptr))
        blocked = np.union1d(rows * self.n + adjacency.indices, np.arange(self.n, dtype=np.int64) * (self.n + 1))
        self.thresholds = blocked - np.arange(len(blocked))
        self.available = self.n ** 2 - len(blocked)

    def draw(self, rng, budget, negatives):
        positives = self.positives
        if budget < len(positives):
            positives = positives[rng.choice(len(positives), budget, replace=False)]
        if not self.available:
            return positives, np.empty((0, 2), dtype=np.int64)
        ranks = rng.integers(self.available, size=len(positives) * negatives)
        keys = ranks + np.searchsorted(self.thresholds, ranks, side="right")
        return positives, np.column_stack(divmod(keys, self.n))


def autoencoder_coordinates(dataset, args, stage):
    adjacency = dataset.adjacency(False)
    pairs = ReconstructionPairs(adjacency)
    details = {"algorithm": "two-layer featureless GCN graph autoencoder", "decoder": "inner product",
               "direction": "undirected", "hidden_width": args.gae_hidden, "positive_pairs": len(pairs.positives),
               "available_ordered_nonedges": pairs.available, "positive_budget_per_epoch": args.gae_edge_budget,
               "negatives_per_positive": args.gae_negatives, "learning_rate": args.gae_lr, "epochs_completed": 0}
    if not len(pairs.positives):
        return np.zeros((dataset.num_nodes, args.embedding_dim), dtype=np.float32), details
    graph = normalized_graph(adjacency)
    rng = np.random.default_rng(args.seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        model = GraphAutoencoder(dataset.num_nodes, args.gae_hidden, args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gae_lr)
    checkpoint = stage / "progress.pt"
    completed = 0
    if checkpoint.is_file():
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        completed = saved["epoch"]
    for epoch in range(completed + 1, args.gae_epochs + 1):
        positives, negatives = pairs.draw(rng, args.gae_edge_budget, args.gae_negatives)
        optimizer.zero_grad(set_to_none=True)
        points = model(graph)
        def logits(edges):
            ids = torch.from_numpy(edges)
            return (points[ids[:, 0]] * points[ids[:, 1]]).sum(dim=1)
        loss = F.softplus(-logits(positives)).mean()
        if len(negatives):
            loss = .5 * (loss + F.softplus(logits(negatives)).mean())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite graph reconstruction loss")
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.gae_epochs or epoch % args.checkpoint_interval == 0:
            atomic_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "rng": rng.bit_generator.state}, checkpoint)
    with torch.no_grad():
        values = model(graph).numpy()
    details["epochs_completed"] = args.gae_epochs
    return values, details
