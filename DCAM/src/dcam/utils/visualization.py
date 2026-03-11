"""Visualization helpers for decoded prototypes and clustering diagnostics."""

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _label_palette(num_labels: int):
    if num_labels <= 10:
        return plt.get_cmap("tab10")
    if num_labels <= 20:
        return plt.get_cmap("tab20")
    return plt.get_cmap("gist_ncar")


def _plot_label_scatter(
    ax,
    points_2d: np.ndarray,
    labels: np.ndarray,
    centroids_2d: np.ndarray,
    title: str,
) -> None:
    valid = labels >= 0
    unique_labels = np.unique(labels[valid]) if np.any(valid) else np.array([], dtype=np.int64)
    cmap = _label_palette(max(len(unique_labels), 1))

    if unique_labels.size == 0:
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=4, alpha=0.7, c="steelblue", linewidths=0)
    else:
        for index, label in enumerate(unique_labels.tolist()):
            mask = labels == label
            ax.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                s=4,
                alpha=0.65,
                color=cmap(index),
                linewidths=0,
                label=str(label),
            )
        if unique_labels.size <= 20:
            ax.legend(title="label", fontsize=8, title_fontsize=9, loc="best", frameon=False)

    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        s=160,
        c="black",
        marker="X",
        edgecolors="white",
        linewidths=1.0,
        label="centroid",
    )
    for cluster_idx, centroid in enumerate(centroids_2d):
        ax.text(centroid[0], centroid[1], str(cluster_idx), fontsize=9, color="black", ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def _plot_cluster_scatter(
    ax,
    points_2d: np.ndarray,
    clusters: np.ndarray,
    centroids_2d: np.ndarray,
    title: str,
) -> None:
    unique_clusters = np.unique(clusters)
    cmap = _label_palette(max(len(unique_clusters), 1))

    for index, cluster_id in enumerate(unique_clusters.tolist()):
        mask = clusters == cluster_id
        ax.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            s=4,
            alpha=0.65,
            color=cmap(index),
            linewidths=0,
            label=f"cluster {cluster_id}",
        )
    if unique_clusters.size <= 20:
        ax.legend(title="pred cluster", fontsize=8, title_fontsize=9, loc="best", frameon=False)

    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        s=160,
        c="black",
        marker="X",
        edgecolors="white",
        linewidths=1.0,
        label="centroid",
    )
    for cluster_idx, centroid in enumerate(centroids_2d):
        ax.text(centroid[0], centroid[1], str(cluster_idx), fontsize=9, color="black", ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def save_image_grid(images: np.ndarray, path: str | Path, num_channels: int) -> None:
    """Save a square grid of decoded prototypes."""
    k = images.shape[0]
    cols = int(ceil(sqrt(k)))
    rows = int(ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, image in enumerate(images):
        r, c = divmod(idx, cols)
        img = np.transpose(image, (1, 2, 0))
        if num_channels == 1:
            axes[r, c].imshow(img[..., 0], cmap="gray")
        else:
            axes[r, c].imshow(img)
        axes[r, c].set_title(f"ρ[{idx}]", fontsize=10)
        axes[r, c].axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_tsne_label_scatter(
    points_2d: np.ndarray,
    labels: np.ndarray,
    centroids_2d: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    _plot_label_scatter(
        ax=ax,
        points_2d=points_2d,
        labels=labels,
        centroids_2d=centroids_2d,
        title=title,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_tsne_cluster_scatter(
    points_2d: np.ndarray,
    clusters: np.ndarray,
    centroids_2d: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    _plot_cluster_scatter(
        ax=ax,
        points_2d=points_2d,
        clusters=clusters,
        centroids_2d=centroids_2d,
        title=title,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if matrix.shape[1] <= 20:
        ax.set_xticks(np.arange(matrix.shape[1]))
    if matrix.shape[0] <= 20:
        ax.set_yticks(np.arange(matrix.shape[0]))

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
