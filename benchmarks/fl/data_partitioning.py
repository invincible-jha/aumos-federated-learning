"""Data partitioning utilities for FL benchmark reproducibility.

Provides IID and non-IID (Dirichlet) data partitioning following the
experimental setups from McMahan et al. (2017) and Li et al. (2020).

References:
    McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks
        from Decentralized Data. arXiv:1602.05629.
    Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks.
        arXiv:1812.06127.
"""

from __future__ import annotations

import numpy as np
from typing import Any


def iid_partition(
    num_samples: int,
    num_clients: int,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Partition sample indices IID (uniform random) across clients.

    Args:
        num_samples: Total number of data samples.
        num_clients: Number of clients to distribute to.
        rng: Optional random number generator for reproducibility.

    Returns:
        List of index lists, one per client.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    indices = rng.permutation(num_samples).tolist()
    shard_size = num_samples // num_clients
    return [
        indices[i * shard_size : (i + 1) * shard_size]
        for i in range(num_clients)
    ]


def dirichlet_partition(
    labels: list[int] | np.ndarray[Any, Any],
    num_clients: int,
    alpha: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Partition sample indices using Dirichlet distribution for non-IID data.

    Dirichlet(alpha) partitioning replicates the heterogeneous data distribution
    from Li et al. (2020). Lower alpha = more heterogeneous (each client holds
    predominantly one class). alpha=100 approximates IID.

    Args:
        labels: Array of class labels (0-indexed).
        num_clients: Number of clients.
        alpha: Dirichlet concentration parameter.
            - alpha < 1: High heterogeneity (each client has few classes)
            - alpha = 1: Moderate heterogeneity
            - alpha > 10: Approaches IID
        rng: Optional random number generator.

    Returns:
        List of index lists (one per client), with non-IID class distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    labels_array = np.asarray(labels)
    num_classes = int(labels_array.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels_array == class_id)[0].tolist()
        rng.shuffle(class_indices)  # type: ignore[arg-type]

        # Sample Dirichlet proportions for this class
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        # Ensure proportions sum to exactly 1
        proportions = proportions / proportions.sum()

        # Assign samples to clients according to proportions
        boundaries = (np.cumsum(proportions) * len(class_indices)).astype(int)
        boundaries = np.clip(boundaries, 0, len(class_indices))
        boundaries[-1] = len(class_indices)

        start = 0
        for client_id, end in enumerate(boundaries):
            client_indices[client_id].extend(class_indices[start:end])
            start = end

    # Shuffle each client's indices
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])  # type: ignore[arg-type]

    return client_indices


def pathological_partition(
    labels: list[int] | np.ndarray[Any, Any],
    num_clients: int,
    num_classes_per_client: int = 2,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Pathological non-IID partition: each client holds exactly k classes.

    Follows the MNIST pathological setup from McMahan et al. (2017) where
    each of 100 clients holds data from exactly 2 classes (200 samples each).

    Args:
        labels: Array of class labels.
        num_clients: Number of clients.
        num_classes_per_client: Number of distinct classes per client (k).
        rng: Optional random number generator.

    Returns:
        List of index lists (one per client).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    labels_array = np.asarray(labels)
    num_classes = int(labels_array.max()) + 1

    # Sort indices by label to create class shards
    sorted_indices = np.argsort(labels_array).tolist()
    num_shards = num_clients * num_classes_per_client
    shard_size = len(sorted_indices) // num_shards

    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size]
        for i in range(num_shards)
    ]

    # Shuffle shard assignment
    shard_assignment = rng.permutation(num_shards).tolist()

    client_indices: list[list[int]] = []
    for client_id in range(num_clients):
        assigned: list[int] = []
        for j in range(num_classes_per_client):
            shard_id = shard_assignment[client_id * num_classes_per_client + j]
            assigned.extend(shards[shard_id])
        client_indices.append(assigned)

    del num_classes  # suppress unused
    return client_indices


def compute_heterogeneity_stats(
    client_indices: list[list[int]],
    labels: list[int] | np.ndarray[Any, Any],
) -> dict[str, float]:
    """Compute data heterogeneity statistics across clients.

    Args:
        client_indices: Partitioned indices (one list per client).
        labels: Full label array.

    Returns:
        Dict with keys: mean_class_entropy, std_class_entropy, min_samples,
        max_samples, mean_samples.
    """
    import math

    labels_array = np.asarray(labels)
    num_classes = int(labels_array.max()) + 1
    entropies: list[float] = []

    for indices in client_indices:
        if not indices:
            continue
        client_labels = labels_array[indices]
        counts = np.bincount(client_labels, minlength=num_classes).astype(float)
        probs = counts / counts.sum()
        # Shannon entropy
        entropy = -float(sum(p * math.log2(p + 1e-12) for p in probs if p > 0))
        entropies.append(entropy)

    sizes = [len(idx) for idx in client_indices if idx]

    return {
        "mean_class_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "std_class_entropy": float(np.std(entropies)) if entropies else 0.0,
        "min_samples": float(min(sizes)) if sizes else 0.0,
        "max_samples": float(max(sizes)) if sizes else 0.0,
        "mean_samples": float(np.mean(sizes)) if sizes else 0.0,
    }
