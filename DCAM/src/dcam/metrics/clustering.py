"""Clustering and reconstruction metrics."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score



def reconstruction_loss_numpy(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean((x - x_hat) ** 2))



def silhouette_safe(z: np.ndarray, c: np.ndarray, max_samples: int | None = None) -> float:
    unique = np.unique(c)
    if unique.size < 2:
        return float("nan")
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



def maybe_supervised_scores(y_true: np.ndarray, c: np.ndarray) -> dict[str, float | None]:
    valid = y_true >= 0
    if not np.any(valid):
        return {"nmi": None, "ari": None}
    return {
        "nmi": float(normalized_mutual_info_score(y_true[valid], c[valid])),
        "ari": float(adjusted_rand_score(y_true[valid], c[valid])),
    }
