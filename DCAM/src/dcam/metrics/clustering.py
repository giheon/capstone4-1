"""Clustering and reconstruction metrics."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def silhouette_safe(z: np.ndarray, c: np.ndarray, max_samples: int | None = None) -> float:
    unique = np.unique(c)
    if unique.size < 2:
        raise ValueError(
            "Silhouette score is undefined because the predicted clusters collapsed to a single label."
        )
    if unique.size >= len(z):
        raise ValueError(
            "Silhouette score is undefined because every sample was assigned to its own cluster."
        )
    sample_size = None
    if max_samples is not None and len(z) > max_samples:
        sample_size = max_samples
    return float(silhouette_score(z, c, sample_size=sample_size))


def entropy_of_clusters(c: np.ndarray) -> float:
    counts = Counter(c.tolist())
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def cluster_size_range(c: np.ndarray) -> tuple[int, int]:
    counts = Counter(c.tolist())
    values = list(counts.values())
    return max(values), min(values)


def cluster_label_weight(
    y_true: np.ndarray,
    c: np.ndarray,
    num_clusters: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = y_true >= 0
    if not np.any(valid):
        k = 0 if num_clusters is None else int(num_clusters)
        return np.zeros((k, 0), dtype=np.int64), np.asarray([], dtype=np.int64)

    labels = np.unique(y_true[valid]).astype(np.int64)
    label_to_index = {int(label): idx for idx, label in enumerate(labels.tolist())}
    k = int(num_clusters) if num_clusters is not None else int(np.max(c[valid])) + 1
    weight = np.zeros((k, labels.size), dtype=np.int64)

    for pred_cluster, label in zip(c[valid].astype(np.int64), y_true[valid].astype(np.int64), strict=True):
        if 0 <= pred_cluster < k:
            weight[pred_cluster, label_to_index[int(label)]] += 1

    return weight, labels


def clustering_accuracy(y_true: np.ndarray, c: np.ndarray, num_clusters: int | None = None) -> float | None:
    weight, _ = cluster_label_weight(y_true=y_true, c=c, num_clusters=num_clusters)
    if weight.size == 0 or weight.shape[1] == 0:
        return None

    max_weight = int(weight.max()) if weight.size > 0 else 0
    row_ind, col_ind = linear_sum_assignment(max_weight - weight)
    matched = int(weight[row_ind, col_ind].sum())
    total = int(weight.sum())
    if total == 0:
        return None
    return float(matched / total)


def maybe_supervised_scores(y_true: np.ndarray, c: np.ndarray) -> dict[str, float | None]:
    valid = y_true >= 0
    if not np.any(valid):
        return {"nmi": None, "ari": None, "acc": None}
    return {
        "nmi": float(normalized_mutual_info_score(y_true[valid], c[valid])),
        "ari": float(adjusted_rand_score(y_true[valid], c[valid])),
        "acc": clustering_accuracy(y_true=y_true, c=c),
    }
