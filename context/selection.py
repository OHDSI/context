from dataclasses import dataclass

import numpy as np

from context.retrieval import edge_index, query_rows, rank_neighbors


@dataclass
class GraphPartition:
    training: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_targets: object
    test_targets: object
    validation_known: object
    test_known: object
    validation_queries: np.ndarray
    test_queries: np.ndarray
    direction: str

    def score(self, model, split):
        if split == "validation":
            targets, known, queries = self.validation_targets, self.validation_known, self.validation_queries
        elif split == "test":
            targets, known, queries = self.test_targets, self.test_known, self.test_queries
        else:
            raise ValueError("Expected validation or test edges")
        curvature = model.curvature.item() if model.curvature is not None else None
        return rank_neighbors(model.coordinates().detach().cpu().numpy(), targets, queries, curvature, known)


def split_edges(dataset, args):
    edges = dataset.edges_list.cpu().numpy()
    source, target = edges.T
    if not args.directed:
        keys = np.minimum(source, target) * dataset.num_nodes + np.maximum(source, target)
    else:
        keys = source * dataset.num_nodes + target
    groups, group_for_edge = np.unique(keys, return_inverse=True)
    count = max(1, int(len(groups) * args.validation_fraction))
    if len(groups) <= 2 * count:
        raise ValueError("Insufficient independent edges for training, validation and test")
    assignments = np.zeros(len(groups), dtype=np.uint8)
    order = np.random.default_rng(args.split_seed).permutation(len(groups))
    if args.split_policy == "preserve_degrees":
        shuffled_edges = np.random.default_rng(args.split_seed).permutation(len(edges))
        protected = np.zeros(len(groups), dtype=bool)
        for endpoints in (source, target):
            _, first = np.unique(endpoints[shuffled_edges], return_index=True)
            protected[group_for_edge[shuffled_edges[first]]] = True
        order = order[~protected[order]]
        if len(order) < 2 * count:
            raise ValueError("Insufficient redundant edges after protecting one edge per nonzero in-degree and out-degree")
    assignments[order[:count]] = 1
    assignments[order[count:2 * count]] = 2
    folds = assignments[group_for_edge]
    training, validation, test = (edges[folds == fold] for fold in range(3))
    def retrieval_index(pairs):
        if args.retrieval_direction == "parents":
            pairs = pairs[:, ::-1]
        elif args.retrieval_direction == "observed" and not args.directed:
            pairs = np.concatenate((pairs, pairs[:, ::-1]))
        return edge_index(pairs, dataset.num_nodes)
    validation_targets = retrieval_index(validation)
    test_targets = retrieval_index(test)
    partition = GraphPartition(training, validation, test, validation_targets, test_targets,
                               retrieval_index(training), retrieval_index(np.concatenate((training, validation))),
                               query_rows(validation_targets, args.eval_queries, args.query_seed),
                               query_rows(test_targets, args.eval_queries, args.query_seed), args.retrieval_direction)
    return partition
