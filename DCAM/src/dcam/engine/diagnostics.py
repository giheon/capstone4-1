"""t-SNE and assignment diagnostics for DCAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from dcam.engine.evaluate import collect_outputs
from dcam.metrics.clustering import cluster_label_weight
from dcam.utils.io import ensure_dir, save_json, save_numpy
from dcam.utils.visualization import save_heatmap, save_tsne_cluster_scatter, save_tsne_label_scatter
from dcam.utils.wandb import log_wandb_images


def _resolve_stage_name(epoch: int, total_epochs: int) -> str:
    if epoch <= 0:
        return "init"
    if epoch >= total_epochs:
        return "final"
    return f"epoch_{epoch:04d}"


def should_run_epoch_diagnostics(epoch: int, total_epochs: int, interval: int) -> bool:
    if epoch >= total_epochs:
        return True
    return interval > 0 and epoch % interval == 0


def _nearest_rho_assignments(latent: np.ndarray, rho: np.ndarray) -> np.ndarray:
    diff = latent[:, None, :] - rho[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return dist2.argmin(axis=1).astype(np.int64)


def _resolve_latent_state(
    *,
    payload: dict[str, np.ndarray],
    rho: np.ndarray,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    use_pre_am_latent = epoch <= 0
    latent = payload["v"] if use_pre_am_latent else payload["v_prime"]
    clusters = _nearest_rho_assignments(latent=latent, rho=rho) if use_pre_am_latent else payload["c"].astype(np.int64)
    latent_space = "v" if use_pre_am_latent else "v_prime"
    return latent, clusters, latent_space


def _compute_tsne_embedding(latent: np.ndarray, rho: np.ndarray, logging_cfg) -> tuple[np.ndarray, np.ndarray]:
    combined_latent = np.concatenate([latent, rho], axis=0)
    perplexity_upper_bound = max(1.0, float(combined_latent.shape[0] - 1))
    perplexity = min(
        float(logging_cfg.tsne_perplexity),
        max(1.0, float(combined_latent.shape[0] - 1) / 3.0),
        perplexity_upper_bound,
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=logging_cfg.tsne_learning_rate,
        max_iter=int(logging_cfg.tsne_max_iter),
        init="pca",
        random_state=int(logging_cfg.tsne_random_state),
    )
    embedding = tsne.fit_transform(combined_latent)
    return embedding[: latent.shape[0]], embedding[latent.shape[0] :]


def run_clustering_diagnostics(
    fabric,
    model,
    dataloader,
    T: int,
    epoch: int,
    total_epochs: int,
    output_dir: str | Path,
    logging_cfg,
    wandb_run=None,
) -> None:
    payload = collect_outputs(fabric=fabric, model=model, dataloader=dataloader, T=T)
    if not fabric.is_global_zero:
        return

    diagnostics_root = ensure_dir(Path(output_dir) / "diagnostics")
    stage_name = _resolve_stage_name(epoch=epoch, total_epochs=total_epochs)
    stage_dir = ensure_dir(diagnostics_root / stage_name)

    rho = model.rho.detach().cpu().numpy()
    latent, c, latent_space = _resolve_latent_state(payload=payload, rho=rho, epoch=epoch)
    y = payload["y"].astype(np.int64)

    weight, label_values = cluster_label_weight(y_true=y, c=c, num_clusters=int(model.k))
    label_cluster_distribution = weight.T

    save_numpy(stage_dir / "true_labels.npy", y)
    save_numpy(stage_dir / "pred_clusters.npy", c)
    save_numpy(stage_dir / "rho.npy", rho)
    save_numpy(stage_dir / "cluster_label_weight.npy", weight)
    save_numpy(stage_dir / "label_cluster_distribution.npy", label_cluster_distribution)

    if label_values.size > 0:
        save_json(
            stage_dir / "assignment_summary.json",
            {
                "epoch": epoch,
                "T": T,
                "latent_space": latent_space,
                "labels": label_values.tolist(),
                "num_clusters": int(model.k),
            },
        )

    points_2d, centers_2d = _compute_tsne_embedding(latent=latent, rho=rho, logging_cfg=logging_cfg)

    save_numpy(stage_dir / "tsne_points_2d.npy", points_2d)
    save_numpy(stage_dir / "tsne_centers_2d.npy", centers_2d)

    image_paths: dict[str, Path] = {}

    if np.any(y >= 0):
        label_tsne_path = stage_dir / "tsne_true_labels.png"
        save_tsne_label_scatter(
            points_2d=points_2d,
            labels=y,
            centroids_2d=centers_2d,
            path=label_tsne_path,
            title=f"{stage_name} t-SNE (label-colored)",
        )
        image_paths["diagnostics/tsne_true_labels"] = label_tsne_path

    pred_tsne_path = stage_dir / "tsne_pred_clusters.png"
    save_tsne_cluster_scatter(
        points_2d=points_2d,
        clusters=c,
        centroids_2d=centers_2d,
        path=pred_tsne_path,
        title=f"{stage_name} t-SNE (pred-cluster-colored)",
    )
    image_paths["diagnostics/tsne_pred_clusters"] = pred_tsne_path

    if label_values.size > 0:
        cluster_label_path = stage_dir / "cluster_label_weight_heatmap.png"
        save_heatmap(
            matrix=weight,
            path=cluster_label_path,
            title=f"{stage_name} cluster x label weight",
            x_label="label",
            y_label="cluster",
        )
        image_paths["diagnostics/cluster_label_weight_heatmap"] = cluster_label_path

        label_cluster_path = stage_dir / "label_cluster_distribution.png"
        save_heatmap(
            matrix=label_cluster_distribution,
            path=label_cluster_path,
            title=f"{stage_name} label -> predicted cluster counts",
            x_label="predicted cluster",
            y_label="label",
        )
        image_paths["diagnostics/label_cluster_distribution"] = label_cluster_path

    log_wandb_images(run=wandb_run, images=image_paths, epoch=epoch)
