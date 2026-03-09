"""Optional Weights & Biases helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def init_wandb(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    name: str | None,
    tags: list[str] | None,
    config: dict[str, Any],
    mode: str,
    dir: str | Path | None = None,
):
    if not enabled:
        return None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb logging was requested, but the `wandb` package is not installed."
        ) from exc

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        config=config,
        mode=mode,
        dir=None if dir is None else str(dir),
    )
    _configure_default_axes(run)
    return run


def _configure_default_axes(run) -> None:
    # Keep batch metrics on the default internal step, but render epoch metrics against train/epoch.
    run.define_metric("train/epoch")
    for metric_name in (
        "train/loss",
        "train/recon",
        "train/kld_like",
        "train/z_entropy",
        "train/cat_prior",
        "train/cat_entropy",
        "eval/acc",
        "eval/cluster_purity",
        "eval/label_purity",
        "lr/nn",
        "lr/gmm",
    ):
        run.define_metric(metric_name, step_metric="train/epoch")


def log_wandb_metrics(run, metrics: dict[str, Any], *, step: int | None = None, commit: bool = True) -> None:
    if run is None:
        return
    payload = {key: _normalize_value(value) for key, value in metrics.items()}
    run.log(payload, step=step, commit=commit)


def log_wandb_images(run, images: dict[str, str | Path], *, step: int | None = None, commit: bool = True) -> None:
    if run is None or not images:
        return

    import wandb

    payload = {key: wandb.Image(str(path)) for key, path in images.items()}
    run.log(payload, step=step, commit=commit)


def update_wandb_summary(run, summary: dict[str, Any]) -> None:
    if run is None:
        return
    for key, value in summary.items():
        run.summary[key] = _normalize_value(value)


def finish_wandb(run, *, exit_code: int = 0) -> None:
    if run is None:
        return
    import wandb

    wandb.finish(exit_code=exit_code)
