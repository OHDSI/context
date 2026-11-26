import math

import numpy as np


def validate_walk_settings(args):
    if not 2 <= args.walk_length <= 10000 or not 1 <= args.walk_window < args.walk_length:
        raise ValueError("walk_length must be 2–10000 and walk_window must be positive and smaller than walk_length")
    if min(args.walks_per_node, args.walk_epochs) < 1:
        raise ValueError("walks_per_node and walk_epochs must be positive")
    if args.training_method == "node2vec":
        if args.walk_negatives < 1:
            raise ValueError("node2vec requires a positive walk_negatives setting")
        if not all(math.isfinite(x) and x > 0 and math.isfinite(1 / x) for x in (args.return_p, args.inout_q)):
            raise ValueError("node2vec return_p and inout_q must be finite and positive with finite reciprocals")


class WalkCorpus:
    def __init__(self, adjacency, args):
        self.adjacency = adjacency
        self.degree = np.diff(adjacency.indptr)
        self.active = np.flatnonzero(self.degree)
        self.args = args
        biased = args.training_method == "node2vec" and (args.return_p != 1 or args.inout_q != 1)
        self.keys = None
        if biased:
            rows = np.repeat(np.arange(len(self.degree), dtype=np.int64), self.degree)
            self.keys = rows * len(self.degree) + adjacency.indices
            self.return_weight = 1. / args.return_p
            self.outward_weight = 1. / args.inout_q
            self.envelope = max(1., self.return_weight, self.outward_weight)

    def _uniform(self, current, rng):
        slots = self.adjacency.indptr[current] + rng.integers(self.degree[current])
        return self.adjacency.indices[slots]

    def _weights(self, previous, targets):
        keys = previous * len(self.degree) + targets.astype(np.int64)
        indices = np.searchsorted(self.keys, keys)
        adjacent = self.keys[np.minimum(indices, len(self.keys) - 1)] == keys
        return np.where(targets == previous, self.return_weight, np.where(adjacent, 1., self.outward_weight))

    def step(self, previous, current, rng):
        if self.keys is None or previous is None:
            return self._uniform(current, rng)
        chosen = self.adjacency.indices[self.adjacency.indptr[current]].copy()
        remaining = np.flatnonzero(self.degree[current] > 1)
        for _ in range(8):
            if not len(remaining):
                return chosen
            proposal = self._uniform(current[remaining], rng)
            accepted = rng.random(len(remaining)) < self._weights(previous[remaining], proposal) / self.envelope
            chosen[remaining[accepted]] = proposal[accepted]
            remaining = remaining[~accepted]
        for row in remaining:
            a, b = self.adjacency.indptr[current[row]:current[row] + 2]
            neighbors = self.adjacency.indices[a:b]
            weights = self._weights(int(previous[row]), neighbors)
            weights /= weights.max()
            chosen[row] = rng.choice(neighbors, p=weights / weights.sum())
        return chosen

    def __iter__(self):
        rng = np.random.default_rng(self.args.seed)
        for _ in range(self.args.walks_per_node):
            order = rng.permutation(self.active)
            for start in range(0, len(order), 2048):
                initial = order[start:start + 2048]
                walks = np.empty((len(initial), self.args.walk_length), dtype=np.int64)
                walks[:, 0] = initial
                for i in range(1, self.args.walk_length):
                    previous = walks[:, i - 2] if i > 1 else None
                    walks[:, i] = self.step(previous, walks[:, i - 1], rng)
                yield from walks.tolist()


def walk_coordinates(dataset, args, stage):
    try:
        import gensim
        from gensim.models import Word2Vec
    except ImportError as error:
        raise ImportError("Install the walks extra to use DeepWalk or node2vec: uv sync --extra walks") from error
    corpus = WalkCorpus(dataset.adjacency(False), args)
    hs = args.training_method == "deepwalk"
    result = np.zeros((dataset.num_nodes, args.embedding_dim), dtype=np.float32)
    settings = {"algorithm": "DeepWalk" if hs else "node2vec", "gensim_version": gensim.__version__,
                "objective": "skip-gram hierarchical softmax" if hs else "skip-gram negative sampling",
                "walk_direction": "undirected", "walks_per_node": args.walks_per_node, "walk_length": args.walk_length,
                "walk_window": args.walk_window, "return_p": 1. if hs else args.return_p, "inout_q": 1. if hs else args.inout_q,
                "walk_negatives": 0 if hs else args.walk_negatives, "walk_workers": args.threads or 1,
                "isolates_without_walks": dataset.num_nodes - len(corpus.active), "walk_epochs_completed": 0}
    if not len(corpus.active):
        return result, settings
    checkpoint = stage / "walk.model"
    if checkpoint.is_file():
        model = Word2Vec.load(str(checkpoint))
    else:
        model = Word2Vec(vector_size=args.embedding_dim, window=args.walk_window, min_count=1, sg=1,
                         hs=int(hs), negative=settings["walk_negatives"], sample=0, seed=args.seed,
                         workers=settings["walk_workers"], sorted_vocab=1)
        model.build_vocab(corpus)
        model.context_walk_epoch = 0
    for epoch in range(model.context_walk_epoch + 1, args.walk_epochs + 1):
        span = .025 - .0001
        model.train(corpus, total_examples=model.corpus_count, epochs=1, start_alpha=.025 - span * (epoch - 1) / args.walk_epochs,
                    end_alpha=.025 - span * epoch / args.walk_epochs)
        model.context_walk_epoch = epoch
        if epoch == 1 or epoch == args.walk_epochs or epoch % args.checkpoint_interval == 0:
            temporary = checkpoint.with_suffix(".pending")
            model.save(str(temporary), separately=[])
            temporary.replace(checkpoint)
    result[np.asarray(model.wv.index_to_key, dtype=np.intp)] = model.wv.vectors
    settings["walk_epochs_completed"] = model.context_walk_epoch
    return result, settings
