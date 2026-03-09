"""Clustering metrics and helpers."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-10


def cluster_label_weight(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred and y_true size mismatch")

    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    dim = int(max(y_pred.max(), y_true.max()) + 1)
    weight = np.zeros((dim, dim), dtype=np.int64)

    for index in range(y_pred.size):
        weight[y_pred[index], y_true[index]] += 1

    return weight


def cluster_purity(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.int64)
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D array")

    cluster_totals = np.sum(weight, axis=1)
    dominant_label_counts = np.max(weight, axis=1) if weight.shape[1] > 0 else np.zeros(weight.shape[0], dtype=np.int64)
    purity_per_cluster = np.divide(
        dominant_label_counts,
        cluster_totals,
        out=np.zeros(weight.shape[0], dtype=np.float64),
        where=cluster_totals > 0,
    )
    return float(np.mean(purity_per_cluster)) if purity_per_cluster.size > 0 else 0.0


def label_purity(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.int64)
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D array")

    label_totals = np.sum(weight, axis=0)
    active_labels = label_totals > 0
    if not np.any(active_labels):
        return 0.0

    dominant_cluster_counts = np.max(weight, axis=0) if weight.shape[0] > 0 else np.zeros(weight.shape[1], dtype=np.int64)
    purity_per_label = np.divide(
        dominant_cluster_counts,
        label_totals,
        out=np.zeros(weight.shape[1], dtype=np.float64),
        where=active_labels,
    )
    return float(np.mean(purity_per_label[active_labels]))


def cluster_acc(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    weight = cluster_label_weight(y_pred=y_pred, y_true=y_true)

    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    acc = float(weight[row_ind, col_ind].sum()) / float(y_pred.size)
    assignment = np.stack([row_ind, col_ind], axis=1)
    return acc, assignment, weight


def posterior_numpy(
    z: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)

    n_centroid, latent_dim = mu.shape
    z_expand = np.repeat(z.reshape(1, latent_dim), n_centroid, axis=0)

    log_prob = np.log(theta + EPS)
    log_prob = log_prob - 0.5 * np.sum(np.log(2.0 * math.pi * var + EPS), axis=1)
    log_prob = log_prob - 0.5 * np.sum((z_expand - mu) ** 2 / (var + EPS), axis=1)
    log_prob = log_prob - np.max(log_prob)

    prob = np.exp(log_prob)
    return prob / np.sum(prob)


def reverse_assignment(assignment: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for cluster_id, label_id in assignment:
        mapping[int(label_id)] = int(cluster_id)
    return mapping
