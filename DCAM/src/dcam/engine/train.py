# DCAM fine-tuning loop

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import Fabric

from dcam.engine.diagnostics import run_clustering_diagnostics, should_run_epoch_diagnostics
from dcam.engine.evaluate import evaluate_model
from dcam.utils.io import save_json, save_torch
from dcam.utils.reconstruction import reconstruction_loss_torch
from dcam.utils.wandb import log_wandb_metrics


@dataclass(slots=True)
class CurriculumState:
    T: int
    lr_reduction_count: int = 0


def _maybe_step_scheduler(scheduler, metric: float) -> bool:
    before = [group["lr"] for group in scheduler.optimizer.param_groups]
    scheduler.step(metric)
    after = [group["lr"] for group in scheduler.optimizer.param_groups]
    return any(a < b for a, b in zip(after, before, strict=True))


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "nan"
    value = float(value)
    if value != value:
        return "nan"
    return f"{value:.4f}"


def _as_rankable_score(value: float | int | None) -> float:
    if value is None:
        return float("-inf")
    value = float(value)
    if value != value:
        return float("-inf")
    return value


def _relative_reconstruction_loss(current_rl: float, pretrained_rl: float) -> float:
    if pretrained_rl <= 0.0:
        raise ValueError(f"pretrained_rl must be positive, got {pretrained_rl:.6f}")
    return float((current_rl - pretrained_rl) / pretrained_rl)


def _build_epoch_payload(
    *,
    epoch: int,
    train_loss: float | None,
    T: int,
    lr_e: float,
    lr_d: float,
    lr_rho: float,
    eval_metrics: dict[str, float | int | None],
    pretrained_rl: float,
) -> dict[str, Any]:
    rl = float(eval_metrics["rl"])
    return {
        "epoch": epoch,
        "T": T,
        "train_loss": train_loss,
        "lr_e": lr_e,
        "lr_d": lr_d,
        "lr_rho": lr_rho,
        "pretrained_rl": pretrained_rl,
        "rrl": _relative_reconstruction_loss(current_rl=rl, pretrained_rl=pretrained_rl),
        **eval_metrics,
    }


def _run_training_epoch(
    fabric: Fabric,
    model,
    train_loader,
    *,
    T: int,
    opt_e,
    opt_d,
    opt_rho,
) -> float:
    model.train()
    train_losses: list[float] = []

    for batch in train_loader:
        x = fabric.to_device(batch["x"])
        opt_e.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        opt_rho.zero_grad(set_to_none=True)

        out = model(x=x, T=T)
        loss = reconstruction_loss_torch(
            x=out.x,
            decoder_output=out.x_hat_raw,
        )
        fabric.backward(loss)
        opt_e.step()
        opt_d.step()
        opt_rho.step()
        train_losses.append(float(loss.detach().item()))

    return sum(train_losses) / max(len(train_losses), 1)


def _current_learning_rates(opt_e, opt_d, opt_rho) -> tuple[float, float, float]:
    return (
        float(opt_e.param_groups[0]["lr"]),
        float(opt_d.param_groups[0]["lr"]),
        float(opt_rho.param_groups[0]["lr"]),
    )


def _maybe_advance_curriculum(
    *,
    curriculum_state: CurriculumState,
    curriculum_cfg,
    reduced_lrs: tuple[bool, bool, bool],
) -> None:
    # Paper-style curriculum: count LR reductions triggered by train-loss plateaus,
    # then increase T after enough reductions have accumulated.
    if any(reduced_lrs):
        curriculum_state.lr_reduction_count += 1
    if curriculum_state.lr_reduction_count >= int(curriculum_cfg.lr_reductions_before_increase_t):
        curriculum_state.T = min(int(curriculum_cfg.t_max), curriculum_state.T + 1)
        curriculum_state.lr_reduction_count = 0


def _save_best_checkpoint(
    *,
    fabric: Fabric,
    checkpoint_dir: Path,
    model,
    epoch: int,
    T: int,
    payload: dict[str, float | int],
) -> None:
    if not fabric.is_global_zero:
        return
    save_torch(
        checkpoint_dir / "best.pt",
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "T": T,
            "metrics": payload,
        },
    )
    save_json(checkpoint_dir / "best_metrics.json", payload)


def _should_replace_best_checkpoint(
    *,
    payload: dict[str, Any],
    best_sc: float,
    rrl_threshold: float,
) -> bool:
    rrl = float(payload["rrl"])
    sc = _as_rankable_score(payload.get("sc"))
    return rrl <= rrl_threshold and sc > best_sc


def train_dcam(
    fabric: Fabric,
    model,
    train_loader,
    val_loader,
    eval_loader,
    opt_e,
    opt_d,
    opt_rho,
    sch_e,
    sch_d,
    sch_rho,
    epochs: int,
    curriculum_state: CurriculumState,
    pretrained_metrics: dict[str, float | int | None],
    curriculum_cfg,
    checkpoint_cfg,
    logging_cfg,
    output_dir: str | Path,
    logger,
    wandb_run=None,
) -> dict[str, Any]:
    pretrained_rl = float(pretrained_metrics["rl"])
    rrl_threshold = float(checkpoint_cfg.rrl_threshold)
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lr_e, lr_d, lr_rho = _current_learning_rates(opt_e, opt_d, opt_rho)
    best_payload = _build_epoch_payload(
        epoch=0,
        train_loss=None,
        T=int(curriculum_state.T),
        lr_e=lr_e,
        lr_d=lr_d,
        lr_rho=lr_rho,
        eval_metrics=pretrained_metrics,
        pretrained_rl=pretrained_rl,
    )
    best_sc = _as_rankable_score(best_payload.get("sc"))
    _save_best_checkpoint(
        fabric=fabric,
        checkpoint_dir=checkpoint_dir,
        model=model,
        epoch=0,
        T=int(curriculum_state.T),
        payload=best_payload,
    )

    for epoch in range(1, epochs + 1):
        T_epoch = int(curriculum_state.T)
        train_loss = _run_training_epoch(
            fabric=fabric,
            model=model,
            train_loader=train_loader,
            T=T_epoch,
            opt_e=opt_e,
            opt_d=opt_d,
            opt_rho=opt_rho,
        )

        eval_metrics = evaluate_model(
            fabric=fabric,
            model=model,
            dataloader=eval_loader,
            T=T_epoch,
            silhouette_max_samples=int(logging_cfg.silhouette_max_samples),
        )
        lr_e, lr_d, lr_rho = _current_learning_rates(opt_e, opt_d, opt_rho)
        payload = _build_epoch_payload(
            epoch=epoch,
            train_loss=train_loss,
            T=T_epoch,
            lr_e=lr_e,
            lr_d=lr_d,
            lr_rho=lr_rho,
            eval_metrics=eval_metrics,
            pretrained_rl=pretrained_rl,
        )

        logger.info(
            "[dcam] epoch=%d/%d loss=%.6f T=%d lr_drop_count=%d rl=%s rrl=%s nmi=%s ari=%s acc=%s sc=%s lr_e=%.2e lr_d=%.2e lr_rho=%.2e",
            epoch,
            epochs,
            train_loss,
            T_epoch,
            curriculum_state.lr_reduction_count,
            _format_metric(payload["rl"]),
            _format_metric(payload["rrl"]),
            _format_metric(eval_metrics["nmi"]),
            _format_metric(eval_metrics["ari"]),
            _format_metric(eval_metrics["acc"]),
            _format_metric(eval_metrics["sc"]),
            lr_e,
            lr_d,
            lr_rho,
        )
        log_wandb_metrics(
            run=wandb_run,
            epoch=epoch,
            metrics={
                "train/loss": train_loss,
                "train/T": T_epoch,
                "eval/rl": payload["rl"],
                "eval/rrl": payload["rrl"],
                "eval/sc": eval_metrics["sc"],
                "eval/nmi": eval_metrics["nmi"],
                "eval/ari": eval_metrics["ari"],
                "eval/acc": eval_metrics["acc"],
                "lr/encoder": lr_e,
                "lr/decoder": lr_d,
                "lr/rho": lr_rho,
            },
        )

        if should_run_epoch_diagnostics(
            epoch=epoch,
            total_epochs=epochs,
            interval=int(logging_cfg.diagnostics_every_n_epochs),
        ):
            logger.info("[diagnostics] saving stage=%s", "final" if epoch == epochs else f"epoch_{epoch:04d}")
            run_clustering_diagnostics(
                fabric=fabric,
                model=model,
                dataloader=eval_loader,
                T=T_epoch,
                epoch=epoch,
                total_epochs=epochs,
                output_dir=output_dir,
                logging_cfg=logging_cfg,
                wandb_run=wandb_run,
            )

        if _should_replace_best_checkpoint(
            payload=payload,
            best_sc=best_sc,
            rrl_threshold=rrl_threshold,
        ):
            best_payload = payload
            best_sc = _as_rankable_score(payload.get("sc"))
            _save_best_checkpoint(
                fabric=fabric,
                checkpoint_dir=checkpoint_dir,
                model=model,
                epoch=epoch,
                T=T_epoch,
                payload=payload,
            )

        reduced_e = _maybe_step_scheduler(sch_e, train_loss)
        reduced_d = _maybe_step_scheduler(sch_d, train_loss)
        reduced_rho = _maybe_step_scheduler(sch_rho, train_loss)
        _maybe_advance_curriculum(
            curriculum_state=curriculum_state,
            curriculum_cfg=curriculum_cfg,
            reduced_lrs=(reduced_e, reduced_d, reduced_rho),
        )

        if float(train_loss) <= float(curriculum_cfg.loss_floor):
            break

        if int(curriculum_state.T) >= int(curriculum_cfg.t_max):
            break

    return best_payload
