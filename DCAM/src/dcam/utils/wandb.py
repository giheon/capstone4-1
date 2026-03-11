"""Minimal Weights & Biases helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def init_wandb_run(cfg: DictConfig):
    if not bool(cfg.wandb.enabled):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled, but the `wandb` package is not installed. "
            "Install project requirements or `pip install wandb`."
        ) from exc

    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
        name=None if cfg.wandb.name is None else str(cfg.wandb.name),
        tags=list(cfg.wandb.tags),
        mode=str(cfg.wandb.mode),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run.define_metric("epoch")
    run.define_metric("pretrain/epoch")
    for metric_name in (
        "pretrain/train_loss",
        "pretrain/val_loss",
    ):
        run.define_metric(metric_name, step_metric="pretrain/epoch")
    for metric_name in (
        "train/loss",
        "train/T",
        "eval/sc",
        "eval/nmi",
        "eval/ari",
        "eval/acc",
        "lr/encoder",
        "lr/decoder",
        "lr/rho",
    ):
        run.define_metric(metric_name, step_metric="epoch")
    return run


def log_wandb_metrics(run, metrics: dict[str, Any], epoch: int, axis_name: str = "epoch") -> None:
    if run is None:
        return
    payload = {axis_name: epoch, **metrics}
    run.log(payload)


def log_wandb_images(run, images: dict[str, str | Path], epoch: int, axis_name: str = "epoch") -> None:
    if run is None:
        return

    import wandb

    payload = {axis_name: epoch}
    for key, value in images.items():
        payload[key] = wandb.Image(str(value))
    run.log(payload)


def finish_wandb_run(run) -> None:
    if run is not None:
        run.finish()
