"""Training diagnostics for GMM initialization and latent-space visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from vade.engine.evaluate import encode_dataset, predict_gamma
from vade.metrics.clustering import cluster_label_weight, cluster_purity, label_purity
from vade.utils.io import ensure_dir, save_json, save_numpy
from vade.utils.wandb import log_wandb_images, log_wandb_metrics


EPS = 1e-10


@dataclass(slots=True)
class DiagnosticsConfig:
    enabled: bool = False
    interval_epochs: int = 50
    tsne_interval_epochs: int = 100
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float | str = "auto"
    tsne_max_iter: int = 1000
    tsne_random_state: int = 42


@dataclass(slots=True)
class DiagnosticSnapshot:
    stage: str
    stage_dir: Path
    tsne_points_2d: np.ndarray | None
    tsne_centroids_2d: np.ndarray | None
    labels: np.ndarray


def _entropy_from_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.clip(np.sum(probs), a_min=EPS, a_max=None)
    return float(-(probs * np.log(probs + EPS)).sum())


def _pairwise_centroid_distances(mu_c: np.ndarray) -> np.ndarray:
    diff = mu_c[:, None, :] - mu_c[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def _distance_stats(distance_matrix: np.ndarray) -> dict[str, float]:
    num_centroids = distance_matrix.shape[0]
    if num_centroids <= 1:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}

    upper = distance_matrix[np.triu_indices(num_centroids, k=1)]
    return {
        "min": float(np.min(upper)),
        "max": float(np.max(upper)),
        "mean": float(np.mean(upper)),
    }


def _coerce_learning_rate(value: float | str) -> float | str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "auto"
        return float(lowered)
    return value


def _compute_tsne_with_centroids(
    embeddings: np.ndarray,
    mu_c: np.ndarray,
    config: DiagnosticsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    all_points = np.concatenate([embeddings, mu_c], axis=0).astype(np.float32)
    tsne = TSNE(
        n_components=2,
        perplexity=float(config.tsne_perplexity),
        learning_rate=_coerce_learning_rate(config.tsne_learning_rate),
        max_iter=int(config.tsne_max_iter),
        init="pca",
        random_state=int(config.tsne_random_state),
    )
    all_2d = tsne.fit_transform(all_points).astype(np.float32)
    num_points = embeddings.shape[0]
    return all_2d[:num_points], all_2d[num_points:]


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


def _save_theta_bar(theta: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(theta)), theta, color="cornflowerblue")
    ax.set_title(title)
    ax.set_xlabel("cluster")
    ax.set_ylabel("theta")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_cluster_count_bar(counts: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(counts)), counts, color="salmon")
    ax.set_title(title)
    ax.set_xlabel("cluster")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_matrix_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
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


def _save_tsne_scatter(
    points_2d: np.ndarray,
    labels: np.ndarray,
    centroids_2d: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    _plot_label_scatter(ax=ax, points_2d=points_2d, labels=labels, centroids_2d=centroids_2d, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_tsne_cluster_scatter(
    points_2d: np.ndarray,
    clusters: np.ndarray,
    centroids_2d: np.ndarray,
    path: Path,
    title: str,
) -> None:
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


def _upsert_history_array(path: Path, array: np.ndarray, index: int) -> None:
    array = np.asarray(array)
    if path.exists():
        history = np.load(path)
        if history.ndim == array.ndim:
            history = history[np.newaxis, ...]
        if index < history.shape[0]:
            history[index] = array
        elif index == history.shape[0]:
            history = np.concatenate([history, array[np.newaxis, ...]], axis=0)
        else:
            raise ValueError(f"cannot write index {index} into history with shape {history.shape}")
    else:
        if index != 0:
            raise ValueError(f"cannot initialize history at non-zero index {index}")
        history = array[np.newaxis, ...]
    save_numpy(path, history)


def append_assignment_history(
    *,
    output_dir: str | Path,
    epoch: int,
    weight: np.ndarray,
    logger=None,
) -> dict[str, Any]:
    """Accumulate epoch-wise assignment matrices and purity scalars into run-level files."""
    root_dir = ensure_dir(output_dir)
    weight = np.asarray(weight, dtype=np.int64)
    label_cluster_counts = weight.T.astype(np.int64)
    cluster_purity_value = cluster_purity(weight)
    label_purity_value = label_purity(weight)

    cluster_label_weight_path = root_dir / "cluster_label_weight.npy"
    label_cluster_counts_path = root_dir / "label_cluster_counts.npy"
    assignment_metrics_path = root_dir / "assignment_metrics.json"

    metrics_payload = {
        "epochs": [],
        "cluster_purity": [],
        "label_purity": [],
    }
    if assignment_metrics_path.exists():
        metrics_payload = json.loads(assignment_metrics_path.read_text(encoding="utf-8"))

    epochs = [int(value) for value in metrics_payload.get("epochs", [])]
    cluster_purity_history = [float(value) for value in metrics_payload.get("cluster_purity", [])]
    label_purity_history = [float(value) for value in metrics_payload.get("label_purity", [])]

    if epoch in epochs:
        history_index = epochs.index(epoch)
        cluster_purity_history[history_index] = float(cluster_purity_value)
        label_purity_history[history_index] = float(label_purity_value)
    else:
        history_index = len(epochs)
        epochs.append(int(epoch))
        cluster_purity_history.append(float(cluster_purity_value))
        label_purity_history.append(float(label_purity_value))

    _upsert_history_array(cluster_label_weight_path, weight, history_index)
    _upsert_history_array(label_cluster_counts_path, label_cluster_counts, history_index)
    save_json(
        assignment_metrics_path,
        {
            "epochs": epochs,
            "cluster_purity": cluster_purity_history,
            "label_purity": label_purity_history,
        },
    )

    if logger is not None:
        logger.info(
            "[assignment:epoch_%04d] cluster_purity=%.4f label_purity=%.4f",
            epoch,
            cluster_purity_value,
            label_purity_value,
        )

    return {
        "epochs": epochs,
        "cluster_purity": float(cluster_purity_value),
        "label_purity": float(label_purity_value),
        "cluster_label_weight_path": cluster_label_weight_path,
        "label_cluster_counts_path": label_cluster_counts_path,
        "assignment_metrics_path": assignment_metrics_path,
    }


def save_assignment_heatmaps(
    *,
    output_dir: str | Path,
    stage: str,
    weight: np.ndarray,
) -> dict[str, Path]:
    """Persist stage-level merge/split heatmaps for the selected diagnostics stage."""
    stage_dir = ensure_dir(Path(output_dir) / stage)
    weight = np.asarray(weight, dtype=np.int64)
    label_cluster_counts = weight.T.astype(np.int64)

    cluster_label_heatmap_path = stage_dir / "cluster_label_weight_heatmap.png"
    label_cluster_heatmap_path = stage_dir / "label_cluster_distribution.png"

    _save_matrix_heatmap(
        matrix=weight,
        path=cluster_label_heatmap_path,
        title=f"{stage} cluster x label weight",
        x_label="label",
        y_label="cluster",
    )
    _save_matrix_heatmap(
        matrix=label_cluster_counts,
        path=label_cluster_heatmap_path,
        title=f"{stage} label -> predicted cluster counts",
        x_label="predicted cluster",
        y_label="label",
    )

    return {
        "cluster_label_heatmap_path": cluster_label_heatmap_path,
        "label_cluster_heatmap_path": label_cluster_heatmap_path,
    }


def _save_tsne_comparison(init_snapshot: DiagnosticSnapshot, final_snapshot: DiagnosticSnapshot, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _plot_label_scatter(
        ax=axes[0],
        points_2d=init_snapshot.tsne_points_2d,
        labels=init_snapshot.labels,
        centroids_2d=init_snapshot.tsne_centroids_2d,
        title="GMM Init",
    )
    _plot_label_scatter(
        ax=axes[1],
        points_2d=final_snapshot.tsne_points_2d,
        labels=final_snapshot.labels,
        centroids_2d=final_snapshot.tsne_centroids_2d,
        title="Final",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_diagnostics(
    *,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device,
    config: DiagnosticsConfig,
    output_dir: str | Path,
    stage: str,
    logger=None,
    wandb_run=None,
    step: int | None = None,
    save_tsne: bool = True,
    commit_wandb: bool = True,
) -> DiagnosticSnapshot:
    stage_dir = ensure_dir(Path(output_dir) / stage)

    embeddings = encode_dataset(model=model, features=features, batch_size=batch_size, device=device)
    theta_t, mu_c_t, lambda_c_t = model.mixture_parameters()
    theta = theta_t.detach().cpu().numpy().astype(np.float32)
    mu_c = mu_c_t.detach().cpu().numpy().astype(np.float32)
    lambda_c = lambda_c_t.detach().cpu().numpy().astype(np.float32)

    gamma = predict_gamma(model=model, features=features, batch_size=batch_size, device=device, use_mean=True)
    hard_assignments = np.argmax(gamma, axis=1).astype(np.int64)
    weight = cluster_label_weight(y_pred=hard_assignments, y_true=labels)
    cluster_purity_value = cluster_purity(weight)
    label_purity_value = label_purity(weight)
    cluster_counts = np.bincount(hard_assignments, minlength=model.n_centroid).astype(np.int64)
    distance_matrix = _pairwise_centroid_distances(mu_c)
    distance_stats = _distance_stats(distance_matrix)
    lambda_cluster_mean = np.mean(lambda_c, axis=1)
    lambda_cluster_min = np.min(lambda_c, axis=1)
    lambda_cluster_max = np.max(lambda_c, axis=1)
    assignment_heatmaps = None
    tsne_points_2d = None
    tsne_centroids_2d = None
    if save_tsne:
        assignment_heatmaps = save_assignment_heatmaps(output_dir=output_dir, stage=stage, weight=weight)
        tsne_points_2d, tsne_centroids_2d = _compute_tsne_with_centroids(
            embeddings=embeddings,
            mu_c=mu_c,
            config=config,
        )

    stats = {
        "stage": stage,
        "num_samples": int(features.shape[0]),
        "theta": {
            "entropy": _entropy_from_probs(theta),
            "min": float(np.min(theta)),
            "max": float(np.max(theta)),
            "mean": float(np.mean(theta)),
        },
        "cluster_counts": {
            "entropy": _entropy_from_probs(cluster_counts.astype(np.float64)),
            "min": int(np.min(cluster_counts)),
            "max": int(np.max(cluster_counts)),
            "mean": float(np.mean(cluster_counts)),
        },
        "assignment": {
            "cluster_purity": float(cluster_purity_value),
            "label_purity": float(label_purity_value),
        },
        "lambda": {
            "global_min": float(np.min(lambda_c)),
            "global_max": float(np.max(lambda_c)),
            "global_mean": float(np.mean(lambda_c)),
            "per_cluster_mean": lambda_cluster_mean.astype(float).tolist(),
            "per_cluster_min": lambda_cluster_min.astype(float).tolist(),
            "per_cluster_max": lambda_cluster_max.astype(float).tolist(),
        },
        "mu_c_pairwise_distance": {
            **distance_stats,
            "matrix_shape": list(distance_matrix.shape),
        },
    }

    theta_bar_path = stage_dir / "theta_bar.png"
    cluster_count_bar_path = stage_dir / "cluster_count_bar.png"
    tsne_label_plot_path = stage_dir / "tsne_label_centroid.png"
    tsne_cluster_plot_path = stage_dir / "tsne_pred_cluster_centroid.png"
    tsne_points_path = stage_dir / "tsne_points_2d.npy"
    tsne_centroids_path = stage_dir / "tsne_centroids_2d.npy"

    _save_theta_bar(theta=theta, path=theta_bar_path, title=f"{stage} theta")
    _save_cluster_count_bar(counts=cluster_counts, path=cluster_count_bar_path, title=f"{stage} cluster counts")
    if save_tsne and tsne_points_2d is not None and tsne_centroids_2d is not None:
        _save_tsne_scatter(
            points_2d=tsne_points_2d,
            labels=labels.astype(np.int64),
            centroids_2d=tsne_centroids_2d,
            path=tsne_label_plot_path,
            title=f"{stage} t-SNE (label-colored)",
        )
        _save_tsne_cluster_scatter(
            points_2d=tsne_points_2d,
            clusters=hard_assignments,
            centroids_2d=tsne_centroids_2d,
            path=tsne_cluster_plot_path,
            title=f"{stage} t-SNE (predicted cluster-colored)",
        )
    else:
        for stale_path in (
            tsne_label_plot_path,
            tsne_cluster_plot_path,
            tsne_points_path,
            tsne_centroids_path,
        ):
            if stale_path.exists():
                stale_path.unlink()

    save_numpy(stage_dir / "theta.npy", theta)
    save_numpy(stage_dir / "mu_c.npy", mu_c)
    save_numpy(stage_dir / "lambda_c.npy", lambda_c)
    save_numpy(stage_dir / "labels.npy", labels.astype(np.int64))
    save_numpy(stage_dir / "hard_assignments.npy", hard_assignments)
    save_numpy(stage_dir / "cluster_counts.npy", cluster_counts)
    save_numpy(stage_dir / "mu_c_pairwise_distance.npy", distance_matrix.astype(np.float32))
    if save_tsne and tsne_points_2d is not None and tsne_centroids_2d is not None:
        save_numpy(tsne_points_path, tsne_points_2d)
        save_numpy(tsne_centroids_path, tsne_centroids_2d)
    save_json(stage_dir / "stats.json", stats)

    if logger is not None:
        logger.info(
            "[diagnostics:%s] theta_max=%.4f theta_entropy=%.4f count_max=%d lambda_mean=%.4f mu_dist_min=%.4f",
            stage,
            stats["theta"]["max"],
            stats["theta"]["entropy"],
            stats["cluster_counts"]["max"],
            stats["lambda"]["global_mean"],
            stats["mu_c_pairwise_distance"]["min"],
        )

    log_wandb_metrics(
        wandb_run,
        {
            f"diagnostics/{stage}/theta_entropy": stats["theta"]["entropy"],
            f"diagnostics/{stage}/theta_min": stats["theta"]["min"],
            f"diagnostics/{stage}/theta_max": stats["theta"]["max"],
            f"diagnostics/{stage}/cluster_count_entropy": stats["cluster_counts"]["entropy"],
            f"diagnostics/{stage}/cluster_count_min": stats["cluster_counts"]["min"],
            f"diagnostics/{stage}/cluster_count_max": stats["cluster_counts"]["max"],
            f"diagnostics/{stage}/cluster_purity": stats["assignment"]["cluster_purity"],
            f"diagnostics/{stage}/label_purity": stats["assignment"]["label_purity"],
            f"diagnostics/{stage}/lambda_global_min": stats["lambda"]["global_min"],
            f"diagnostics/{stage}/lambda_global_max": stats["lambda"]["global_max"],
            f"diagnostics/{stage}/lambda_global_mean": stats["lambda"]["global_mean"],
            f"diagnostics/{stage}/mu_distance_min": stats["mu_c_pairwise_distance"]["min"],
            f"diagnostics/{stage}/mu_distance_max": stats["mu_c_pairwise_distance"]["max"],
            f"diagnostics/{stage}/mu_distance_mean": stats["mu_c_pairwise_distance"]["mean"],
        },
        step=step,
        commit=False,
    )
    image_payload = {
        f"diagnostics/{stage}/theta_bar": theta_bar_path,
        f"diagnostics/{stage}/cluster_count_bar": cluster_count_bar_path,
    }
    if assignment_heatmaps is not None:
        image_payload[f"diagnostics/{stage}/label_cluster_distribution"] = assignment_heatmaps[
            "label_cluster_heatmap_path"
        ]
        image_payload[f"diagnostics/{stage}/cluster_label_weight_heatmap"] = assignment_heatmaps[
            "cluster_label_heatmap_path"
        ]
    if save_tsne and tsne_points_2d is not None and tsne_centroids_2d is not None:
        image_payload[f"diagnostics/{stage}/tsne_label"] = tsne_label_plot_path
        image_payload[f"diagnostics/{stage}/tsne_pred_cluster"] = tsne_cluster_plot_path
    log_wandb_images(
        wandb_run,
        image_payload,
        step=step,
        commit=commit_wandb,
    )

    return DiagnosticSnapshot(
        stage=stage,
        stage_dir=stage_dir,
        tsne_points_2d=tsne_points_2d,
        tsne_centroids_2d=tsne_centroids_2d,
        labels=labels.astype(np.int64),
    )


def save_init_final_comparison(
    *,
    init_snapshot: DiagnosticSnapshot,
    final_snapshot: DiagnosticSnapshot,
    output_dir: str | Path,
    logger=None,
    wandb_run=None,
    step: int | None = None,
    commit_wandb: bool = True,
) -> Path:
    if init_snapshot.tsne_points_2d is None or init_snapshot.tsne_centroids_2d is None:
        raise ValueError("init snapshot does not include t-SNE embeddings")
    if final_snapshot.tsne_points_2d is None or final_snapshot.tsne_centroids_2d is None:
        raise ValueError("final snapshot does not include t-SNE embeddings")

    out_dir = ensure_dir(output_dir)
    comparison_path = out_dir / "tsne_init_vs_final.png"
    _save_tsne_comparison(init_snapshot=init_snapshot, final_snapshot=final_snapshot, path=comparison_path)

    if logger is not None:
        logger.info("[diagnostics:final] saved init vs final comparison: %s", comparison_path)

    log_wandb_images(
        wandb_run,
        {"diagnostics/final/tsne_init_vs_final": comparison_path},
        step=step,
        commit=commit_wandb,
    )
    return comparison_path
