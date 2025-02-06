import torch
from tqdm import tqdm


def evaluate_model(model: torch.nn.Module, dataset, batch_size: int = 32):
    mean_rank, map_score = evaluate_mean_rank_and_map(dataset, model.weight.tensor.detach(), dataset.num_nodes,
                                                      batch_size=batch_size)
    print(f"Mean Rank: {mean_rank}, MAP: {map_score}")
    return mean_rank, map_score

def hyperbolic_distance(u, v, c=1.0, eps=1e-5):
    """
    Computes the Poincaré distance between two sets of points.

    Parameters:
    - u: Tensor of shape [batch_size, embedding_dim]
    - v: Tensor of shape [num_nodes, embedding_dim]

    Returns:
    - distances: Tensor of shape [batch_size, num_nodes], containing the distances between each u[i] and v[j].
    """
    # Calculate norms
    sqrt_c = c ** 0.5
    u_norm_sq = torch.sum(u ** 2, dim=1, keepdim=True)  # Shape: [batch_size, 1]
    v_norm_sq = torch.sum(v ** 2, dim=1, keepdim=True)  # Shape: [num_nodes, 1]

    # Ensure norms are within the ball
    u_norm_sq = torch.clamp(u_norm_sq, max=(1 - eps))
    v_norm_sq = torch.clamp(v_norm_sq, max=(1 - eps))

    # Compute inner product
    u = u.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]
    v = v.unsqueeze(0)  # Shape: [1, num_nodes, embedding_dim]
    diff = u - v  # Shape: [batch_size, num_nodes, embedding_dim]
    diff_norm_sq = torch.sum(diff ** 2, dim=2)  # Shape: [batch_size, num_nodes]

    # Compute denominator
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq).transpose(0, 1)  # Shape: [batch_size, num_nodes]
    denom = denom.clamp(min=eps)

    # Compute the argument of arcosh
    argument = 1 + (2 * c * diff_norm_sq) / denom

    # Ensure argument is >= 1
    argument = torch.clamp(argument, min=1 + eps)

    # Compute the distance
    distances = torch.acosh(argument)

    return distances

def fast_average_precision(y_true, y_score):
    # Sort scores in descending order and get sorted indices
    desc_score_indices = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[desc_score_indices]

    num_positive = y_true.sum().item()
    if num_positive == 0:
        return 0.0
    # Calculate true positives and false positives
    tp_cumsum = torch.cumsum(y_true_sorted, dim=0)
    # Calculate precision at each position where y_true_sorted == 1
    positions = torch.arange(1, len(y_true_sorted) + 1, dtype=torch.float32, device=y_true.device)
    precision_at_k = tp_cumsum / positions

    # Only consider positions where y_true_sorted == 1
    precision_at_positives = precision_at_k[y_true_sorted == 1]
    ap = precision_at_positives.sum() / num_positive
    # Move the result back to CPU if needed and return a Python float
    return ap

def evaluate_mean_rank_and_map(dataset, embeddings, num_nodes, batch_size=128):
    edges_list = dataset.edges_list
    device = embeddings.device
    num_edges = edges_list.size(0)
    adjacency_sets = dataset.generate_adjacency_sets()
    mean_ranks = torch.empty(num_edges, dtype=torch.int64, device=device)
    average_precisions = torch.empty(num_edges, dtype=torch.float32, device=device)

    # Convert embeddings to double precision for numerical stability
    # Choose dtype based on device: use float32 for MPS, else float64
    dtype = torch.float32 if embeddings.device.type == "mps" else torch.float64
    embeddings = embeddings.to(dtype=dtype)

    for i in tqdm(range(0, num_edges, batch_size)):
        batch_edges = edges_list[i:i + batch_size]
        batch_size_actual = batch_edges.size(0)

        # Get source nodes and target nodes
        u_nodes = batch_edges[:, 0].to(device)  # Shape: [batch_size_actual]
        v_nodes = batch_edges[:, 1].to(device)  # Shape: [batch_size_actual]

        u_embeddings = embeddings[u_nodes]  # Shape: [batch_size_actual, embedding_dim]

        all_node_embeddings = embeddings

        distances =  hyperbolic_distance(u_embeddings, all_node_embeddings, c=1.0)

        mask = torch.zeros((batch_size_actual, num_nodes), dtype=torch.bool, device=device)

        for idx in range(batch_size_actual):
            u_int = u_nodes[idx].item()
            v_int = v_nodes[idx].item()
            connected_v = adjacency_sets[u_int]
            mask[idx, list(connected_v)] = True  # Mark connected nodes to exclude
            mask[idx, u_int] = True  # Exclude u_node itself
            mask[idx, v_int] = False  # Include the positive v_node

        valid_distances = distances.clone()
        valid_distances[mask] = float('-inf')  # Set masked positions to -inf for max computation
        max_distance_per_row, _ = valid_distances.max(dim=1, keepdim=True)  # Shape: [batch_size_actual, 1]
        max_distance_plus_one = max_distance_per_row + 1  # Shape: [batch_size_actual, 1]
        max_distance_expanded = max_distance_plus_one.expand(-1,
                                                             distances.size(1))  # Shape: [batch_size_actual, num_nodes]
        distances[mask] = max_distance_expanded[mask]

        # Now compute ranks
        # The lower the distance, the higher the rank (rank 1 is the smallest distance)
        # So we can sort distances and get the indices
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        # For each example, find the rank of the positive v_node
        matches = (sorted_indices == v_nodes.unsqueeze(1))
        ranks = matches.nonzero()[:, 1] + 1  # rank positions start from 1
        mean_ranks[i:i + batch_size_actual] = ranks

        # For MAP, we need to compute average precision for each example
        # Create labels: 1 for positive v_node, 0 for others
        labels = torch.zeros_like(distances, dtype=torch.int64, device=device)
        labels[torch.arange(batch_size_actual, device=device), v_nodes] = 1
        # After sorting, obtain sorted labels
        sorted_labels = torch.gather(labels, 1, sorted_indices)
        # Compute average precision for each u_node
        aps = torch.empty(batch_size_actual, dtype=torch.float32, device=device)
        for idx in range(batch_size_actual):
            ap = fast_average_precision(sorted_labels[idx], -sorted_distances[idx])
            aps[idx] = ap
        average_precisions[i:i + batch_size_actual] = aps

    # Calculate Mean Rank and MAP
    mean_rank = mean_ranks.float().mean().item()
    map_score = average_precisions.mean().item()
    return mean_rank, map_score