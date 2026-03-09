"""Main VaDE training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from vade.config import TrainConfig
from vade.engine.diagnostics import (
    DiagnosticSnapshot,
    DiagnosticsConfig,
    append_assignment_history,
    run_diagnostics,
    save_init_final_comparison,
)
from vade.engine.evaluate import encode_dataset, predict_gamma
from vade.engine.pretrain import initialize_from_pretraining
from vade.metrics.clustering import cluster_acc, cluster_purity, label_purity
from vade.utils.wandb import log_wandb_metrics


EPS = 1e-10
TrainHistory = dict[str, list[Any]]


@dataclass(slots=True)
class EpochSummary:
    loss: float
    recon: float
    kld_like: float
    z_entropy: float
    cat_prior: float
    cat_entropy: float
    acc: float
    cluster_purity: float
    label_purity: float
    lr_nn: float
    lr_gmm: float


def initialize_gmm_parameters(model, embeddings: np.ndarray, config: TrainConfig) -> None:
    """Initialize GMM parameters from latent embeddings before VaDE training."""
    with torch.no_grad():
        model.pi_logits.zero_()
        model.mu_c.zero_()
        model.log_var_c.zero_()

    if config.dataset == "reuters10k":
        kmeans = KMeans(n_clusters=model.n_centroid, random_state=0, n_init=20)
        labels = kmeans.fit_predict(embeddings)

        means = kmeans.cluster_centers_.astype(np.float32)
        weights = np.zeros(model.n_centroid, dtype=np.float32)
        variances = np.zeros((model.n_centroid, model.latent_dim), dtype=np.float32)
        global_var = np.var(embeddings, axis=0).astype(np.float32) + config.gmm_reg_covar

        for cluster_id in range(model.n_centroid):
            members = embeddings[labels == cluster_id]
            if len(members) == 0:
                weights[cluster_id] = 1.0 / model.n_centroid
                variances[cluster_id] = global_var
                continue
            weights[cluster_id] = len(members) / len(embeddings)
            variances[cluster_id] = np.var(members, axis=0).astype(np.float32) + config.gmm_reg_covar

        with torch.no_grad():
            model.mu_c.copy_(torch.from_numpy(means))
            model.log_var_c.copy_(torch.log(torch.from_numpy(variances) + EPS))
            model.pi_logits.copy_(torch.log(torch.from_numpy(weights) + EPS))
        return

    random_state = 3 if config.dataset == "har" else 0
    gmm = GaussianMixture(
        n_components=model.n_centroid,
        covariance_type="diag",
        random_state=random_state,
        n_init=10,
        reg_covar=config.gmm_reg_covar,
    )
    gmm.fit(embeddings)

    with torch.no_grad():
        model.mu_c.copy_(torch.from_numpy(gmm.means_.astype(np.float32)))
        model.log_var_c.copy_(torch.log(torch.from_numpy(gmm.covariances_.astype(np.float32)) + EPS))
        model.pi_logits.copy_(torch.log(torch.from_numpy(gmm.weights_.astype(np.float32)) + EPS))


def lr_decay_step(
    optimizer: torch.optim.Optimizer,
    dataset: str,
    decay_nn: float,
    decay_gmm: float,
) -> tuple[float, float]:
    """Apply dataset-specific LR decay to NN and GMM parameter groups."""
    nn_lr = optimizer.param_groups[0]["lr"]
    gmm_lr = optimizer.param_groups[1]["lr"]

    if dataset == "mnist":
        nn_lr = max(nn_lr * decay_nn, 0.0002)
        gmm_lr = max(gmm_lr * decay_gmm, 0.0002)
    else:
        nn_lr = nn_lr * decay_nn
        gmm_lr = gmm_lr * decay_gmm

    optimizer.param_groups[0]["lr"] = nn_lr
    optimizer.param_groups[1]["lr"] = gmm_lr
    return nn_lr, gmm_lr


def build_history() -> TrainHistory:
    """Create the persistent training history structure stored in checkpoints."""
    return {
        "step": [],
        "epoch": [],
        "step_in_epoch": [],
        "batch_gamma_entropy": [],
        "batch_theta_entropy": [],
        "batch_cluster_usage_entropy": [],
        "batch_cluster_top1_ratio": [],
        "batch_recon": [],
        "batch_kld_like": [],
        "loss": [],
        "recon": [],
        "kld_like": [],
        "z_entropy": [],
        "cat_prior": [],
        "cat_entropy": [],
        "acc": [],
        "cluster_purity": [],
        "label_purity": [],
        "lr_nn": [],
        "lr_gmm": [],
    }


def should_record_training_step(global_step: int) -> bool:
    """Record batch-level diagnostics every 50 optimization steps."""
    return global_step % 50 == 0


def record_training_step(
    history: TrainHistory,
    *,
    global_step: int,
    epoch_index: int,
    step_in_epoch: int,
    logs: dict[str, Any],
    wandb_run=None,
) -> None:
    """Append step-level history and optionally mirror it to W&B."""
    step_metrics = torch.stack(
        [
            logs["gamma_entropy"],
            logs["theta_entropy"],
            logs["cluster_usage_entropy"],
            logs["cluster_top1_ratio"],
            logs["recon"],
            logs["kld_like"],
        ]
    ).detach().cpu().tolist()

    history["step"].append(global_step)
    history["epoch"].append(epoch_index + 1)
    history["step_in_epoch"].append(step_in_epoch)
    history["batch_gamma_entropy"].append(float(step_metrics[0]))
    history["batch_theta_entropy"].append(float(step_metrics[1]))
    history["batch_cluster_usage_entropy"].append(float(step_metrics[2]))
    history["batch_cluster_top1_ratio"].append(float(step_metrics[3]))
    history["batch_recon"].append(float(step_metrics[4]))
    history["batch_kld_like"].append(float(step_metrics[5]))

    log_wandb_metrics(
        wandb_run,
        {
            "train/batch_gamma_entropy": float(step_metrics[0]),
            "train/batch_theta_entropy": float(step_metrics[1]),
            "train/batch_cluster_usage_entropy": float(step_metrics[2]),
            "train/batch_cluster_top1_ratio": float(step_metrics[3]),
            "train/batch_recon": float(step_metrics[4]),
            "train/batch_kld_like": float(step_metrics[5]),
            "train/epoch": epoch_index + 1,
            "train/step_in_epoch": step_in_epoch,
        },
        step=global_step,
    )


def build_epoch_summary(
    epoch_metrics: list[float],
    *,
    acc: float,
    cluster_purity: float,
    label_purity: float,
    lr_nn: float,
    lr_gmm: float,
) -> EpochSummary:
    """Normalize epoch-level training/evaluation metrics into a typed summary."""
    return EpochSummary(
        loss=float(epoch_metrics[0]),
        recon=float(epoch_metrics[1]),
        kld_like=float(epoch_metrics[2]),
        z_entropy=float(epoch_metrics[3]),
        cat_prior=float(epoch_metrics[4]),
        cat_entropy=float(epoch_metrics[5]),
        acc=float(acc),
        cluster_purity=float(cluster_purity),
        label_purity=float(label_purity),
        lr_nn=float(lr_nn),
        lr_gmm=float(lr_gmm),
    )


def record_epoch_summary(
    history: TrainHistory,
    *,
    summary: EpochSummary,
    epoch_index: int,
    global_step: int,
    wandb_run=None,
) -> None:
    """Append epoch-level history and mirror it to W&B."""
    history["loss"].append(summary.loss)
    history["recon"].append(summary.recon)
    history["kld_like"].append(summary.kld_like)
    history["z_entropy"].append(summary.z_entropy)
    history["cat_prior"].append(summary.cat_prior)
    history["cat_entropy"].append(summary.cat_entropy)
    history["acc"].append(summary.acc)
    history["cluster_purity"].append(summary.cluster_purity)
    history["label_purity"].append(summary.label_purity)
    history["lr_nn"].append(summary.lr_nn)
    history["lr_gmm"].append(summary.lr_gmm)

    log_wandb_metrics(
        wandb_run,
        {
            "train/loss": summary.loss,
            "train/recon": summary.recon,
            "train/kld_like": summary.kld_like,
            "train/z_entropy": summary.z_entropy,
            "train/cat_prior": summary.cat_prior,
            "train/cat_entropy": summary.cat_entropy,
            "eval/acc": summary.acc,
            "eval/cluster_purity": summary.cluster_purity,
            "eval/label_purity": summary.label_purity,
            "lr/nn": summary.lr_nn,
            "lr/gmm": summary.lr_gmm,
            "train/epoch": epoch_index + 1,
        },
        step=global_step,
        commit=True,
    )


def _log_info(logger, message: str) -> None:
    if logger is None:
        print(message)
    else:
        logger.info(message)


def _build_optimizer(model, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": model.nn_parameters(), "lr": config.lr_nn},
            {"params": model.gmm_parameters(), "lr": config.lr_gmm},
        ],
        eps=1e-8,
    )


def _build_loader(features: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
    data = torch.from_numpy(features.astype(np.float32))
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)


def _maybe_decay_learning_rates(
    *,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch_index: int,
    logger=None,
) -> None:
    if epoch_index == 0 or epoch_index % config.decay_n != 0:
        return

    lr_nn, lr_gmm = lr_decay_step(
        optimizer=optimizer,
        dataset=config.dataset,
        decay_nn=config.decay_nn,
        decay_gmm=config.decay_gmm,
    )
    _log_info(logger, f"lr_nn: {lr_nn:.8f} | lr_gmm: {lr_gmm:.8f}")


def _run_training_epoch( # train per epoch
    *,
    model,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    history: TrainHistory,
    epoch_index: int,
    global_step: int,
    wandb_run=None,
) -> tuple[list[float], int]:
    model.train()

    run_loss = torch.zeros((), device=device)
    run_recon = torch.zeros((), device=device)
    run_kld_like = torch.zeros((), device=device)
    run_z_entropy = torch.zeros((), device=device)
    run_cat_prior = torch.zeros((), device=device)
    run_cat_entropy = torch.zeros((), device=device)
    n = 0

    for step_in_epoch, batch_x in enumerate(loader, start=1):
        batch_x = batch_x.to(device)
        optimizer.zero_grad(set_to_none=True)

        out = model(batch_x)
        loss, loss_logs = model.vade_loss(
            x=out.x,
            x_hat=out.x_hat,
            z=out.z,
            z_mean=out.z_mean,
            z_log_var=out.z_log_var,
            alpha=config.alpha,
        )
        loss.backward()
        optimizer.step()
        global_step += 1

        if should_record_training_step(global_step):
            record_training_step(
                history,
                global_step=global_step,
                epoch_index=epoch_index,
                step_in_epoch=step_in_epoch,
                logs=loss_logs,
                wandb_run=wandb_run,
            )

        batch_size_now = batch_x.shape[0]
        run_loss += loss.detach() * batch_size_now
        run_recon += loss_logs["recon"] * batch_size_now
        run_kld_like += loss_logs["kld_like"] * batch_size_now
        run_z_entropy += loss_logs["z_entropy"] * batch_size_now
        run_cat_prior += loss_logs["cat_prior"] * batch_size_now
        run_cat_entropy += loss_logs["cat_entropy"] * batch_size_now
        n += batch_size_now

    epoch_metrics = (
        torch.stack([run_loss, run_recon, run_kld_like, run_z_entropy, run_cat_prior, run_cat_entropy])
        / max(n, 1)
    ).detach().cpu().tolist()
    return [float(metric) for metric in epoch_metrics], global_step


def _evaluate_epoch( # eval per epoch
    *,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    eval_use_mean: bool,
    epoch_metrics: list[float],
    optimizer: torch.optim.Optimizer,
) -> tuple[EpochSummary, np.ndarray]:
    gamma = predict_gamma(
        model=model,
        features=features,
        batch_size=config.batch_size,
        device=device,
        use_mean=eval_use_mean,
    )
    y_pred = np.argmax(gamma, axis=1)
    acc, _, weight = cluster_acc(y_pred, labels)

    summary = build_epoch_summary(
        epoch_metrics,
        acc=acc,
        cluster_purity=cluster_purity(weight),
        label_purity=label_purity(weight),
        lr_nn=float(optimizer.param_groups[0]["lr"]),
        lr_gmm=float(optimizer.param_groups[1]["lr"]),
    )
    return summary, weight


def _record_epoch_monitoring(
    *,
    history: TrainHistory,
    summary: EpochSummary,
    epoch_index: int,
    global_step: int,
    weight: np.ndarray,
    diagnostics_config: DiagnosticsConfig,
    diagnostics_dir,
    wandb_run=None,
) -> None:
    if diagnostics_config.enabled and diagnostics_dir is not None:
        append_assignment_history(
            output_dir=diagnostics_dir,
            epoch=epoch_index + 1,
            weight=weight,
            logger=None,
        )

    record_epoch_summary(
        history,
        summary=summary,
        epoch_index=epoch_index,
        global_step=global_step,
        wandb_run=wandb_run,
    )


def _maybe_run_epoch_diagnostics(
    *,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    diagnostics_config: DiagnosticsConfig,
    diagnostics_dir,
    epoch_index: int,
    global_step: int,
    logger=None,
    wandb_run=None,
) -> None:
    if not diagnostics_config.enabled or diagnostics_dir is None:
        return
    if (epoch_index + 1) % diagnostics_config.interval_epochs != 0:
        return

    should_save_tsne = (epoch_index + 1) % diagnostics_config.tsne_interval_epochs == 0 # t-sne save cycle
    diagnostics_step = global_step + 1
    run_diagnostics(
        model=model,
        features=features,
        labels=labels,
        batch_size=config.batch_size,
        device=device,
        config=diagnostics_config,
        output_dir=diagnostics_dir,
        stage=f"epoch_{epoch_index + 1:04d}",
        logger=logger,
        wandb_run=wandb_run,
        step=diagnostics_step,
        save_tsne=should_save_tsne,
        commit_wandb=True,
    )


def _run_final_diagnostics(
    *,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    diagnostics_config: DiagnosticsConfig,
    diagnostics_dir,
    init_snapshot: DiagnosticSnapshot | None,
    global_step: int,
    logger=None,
    wandb_run=None,
) -> None:
    if not diagnostics_config.enabled or diagnostics_dir is None:
        return

    final_diagnostics_step = global_step + 2
    final_snapshot = run_diagnostics(
        model=model,
        features=features,
        labels=labels,
        batch_size=config.batch_size,
        device=device,
        config=diagnostics_config,
        output_dir=diagnostics_dir,
        stage="final",
        logger=logger,
        wandb_run=wandb_run,
        step=final_diagnostics_step,
        save_tsne=True,
        commit_wandb=init_snapshot is None,
    )
    if init_snapshot is not None:
        save_init_final_comparison(
            init_snapshot=init_snapshot,
            final_snapshot=final_snapshot,
            output_dir=Path(diagnostics_dir) / "final",
            logger=logger,
            wandb_run=wandb_run,
            step=final_diagnostics_step,
            commit_wandb=True,
        )


def train_vade(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    load_pretrained_ae: bool = True,
    eval_use_mean: bool | None = None,
    logger=None,
    wandb_run=None,
    diagnostics_config: DiagnosticsConfig | None = None,
    diagnostics_dir=None,
) -> TrainHistory:
    history = build_history()
    eval_use_mean = config.eval_use_mean if eval_use_mean is None else eval_use_mean
    diagnostics_config = DiagnosticsConfig() if diagnostics_config is None else diagnostics_config

    initialize_from_pretraining(
        model=model,
        features=features,
        config=config,
        device=device,
        load_pretrained_ae=load_pretrained_ae,
        logger=logger,
    )
    embeddings = encode_dataset(model=model, features=features, batch_size=config.batch_size, device=device)
    initialize_gmm_parameters(model=model, embeddings=embeddings, config=config)

    init_snapshot = None
    if diagnostics_config.enabled and diagnostics_dir is not None:
        init_snapshot = run_diagnostics(
            model=model,
            features=features,
            labels=labels,
            batch_size=config.batch_size,
            device=device,
            config=diagnostics_config,
            output_dir=diagnostics_dir,
            stage="init",
            logger=logger,
            wandb_run=wandb_run,
            step=0,
            save_tsne=True,
        )

    optimizer = _build_optimizer(model, config)
    loader = _build_loader(features, config.batch_size)
    global_step = 0

    for epoch_index in range(config.epochs):
        _maybe_decay_learning_rates(
            optimizer=optimizer,
            config=config,
            epoch_index=epoch_index,
            logger=logger,
        )

        epoch_metrics, global_step = _run_training_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            config=config,
            device=device,
            history=history,
            epoch_index=epoch_index,
            global_step=global_step,
            wandb_run=wandb_run,
        )

        summary, weight = _evaluate_epoch(
            model=model,
            features=features,
            labels=labels,
            config=config,
            device=device,
            eval_use_mean=eval_use_mean,
            epoch_metrics=epoch_metrics,
            optimizer=optimizer,
        )
        _record_epoch_monitoring(
            history=history,
            summary=summary,
            epoch_index=epoch_index,
            global_step=global_step,
            weight=weight,
            diagnostics_config=diagnostics_config,
            diagnostics_dir=diagnostics_dir,
            wandb_run=wandb_run,
        )

        _log_info(
            logger,
            (
                f"epoch {epoch_index + 1:04d}/{config.epochs} | "
                f"loss={summary.loss:.6f} | "
                f"acc_p_c_z={summary.acc:.6f} | "
                f"lr_nn={summary.lr_nn:.8f} | "
                f"lr_gmm={summary.lr_gmm:.8f}"
            ),
        )

        if epoch_index == 1 and config.dataset == "har" and summary.acc < 0.77:
            raise RuntimeError("HAR dataset bad init (acc < 0.77 at epoch 2). Please run again.")
        if not math.isfinite(summary.loss):
            raise RuntimeError("non-finite VaDE loss encountered")

        _maybe_run_epoch_diagnostics(
            model=model,
            features=features,
            labels=labels,
            config=config,
            device=device,
            diagnostics_config=diagnostics_config,
            diagnostics_dir=diagnostics_dir,
            epoch_index=epoch_index,
            global_step=global_step,
            logger=logger,
            wandb_run=wandb_run,
        )

    _run_final_diagnostics(
        model=model,
        features=features,
        labels=labels,
        config=config,
        device=device,
        diagnostics_config=diagnostics_config,
        diagnostics_dir=diagnostics_dir,
        init_snapshot=init_snapshot,
        global_step=global_step,
        logger=logger,
        wandb_run=wandb_run,
    )
    return history
