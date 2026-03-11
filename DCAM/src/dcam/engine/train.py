# DCAM fine-tuning loop

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def _run_training_epoch(
    fabric: Fabric,
    model,
    train_loader,
    *,
    curriculum_state: CurriculumState,
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

        out = model(x=x, T=curriculum_state.T)
        loss = reconstruction_loss_torch(
            x=out.x,
            decoder_output=out.x_hat_raw,
            loss_type=model.ae.reconstruction_loss,
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
    payload: dict[str, float | int],
    curriculum_state: CurriculumState,
) -> None:
    if not fabric.is_global_zero:
        return
    save_torch(
        checkpoint_dir / "best.pt",
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "T": curriculum_state.T,
            "metrics": payload,
        },
    )
    save_json(checkpoint_dir / "best_metrics.json", payload)


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
    curriculum_cfg,
    logging_cfg,
    output_dir: str | Path,
    logger,
    wandb_run=None,
) -> dict[str, float | int]:
    best_metric = float("inf")
    best_payload: dict[str, float | int] = {}
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = _run_training_epoch(
            fabric=fabric,
            model=model,
            train_loader=train_loader,
            curriculum_state=curriculum_state,
            opt_e=opt_e,
            opt_d=opt_d,
            opt_rho=opt_rho,
        )

        val_metrics = evaluate_model(
            fabric=fabric,
            model=model,
            dataloader=val_loader if len(val_loader) > 0 else eval_loader,
            T=curriculum_state.T,
            silhouette_max_samples=int(logging_cfg.silhouette_max_samples),
        )
        val_rl = float(val_metrics["rl"])

        reduced_e = _maybe_step_scheduler(sch_e, val_rl)
        reduced_d = _maybe_step_scheduler(sch_d, val_rl)
        reduced_rho = _maybe_step_scheduler(sch_rho, val_rl)
        _maybe_advance_curriculum(
            curriculum_state=curriculum_state,
            curriculum_cfg=curriculum_cfg,
            reduced_lrs=(reduced_e, reduced_d, reduced_rho),
        )

        eval_metrics = evaluate_model(
            fabric=fabric,
            model=model,
            dataloader=eval_loader,
            T=curriculum_state.T,
            silhouette_max_samples=int(logging_cfg.silhouette_max_samples),
        )
        lr_e, lr_d, lr_rho = _current_learning_rates(opt_e, opt_d, opt_rho)
        payload = {
            "epoch": epoch,
            "T": curriculum_state.T,
            "train_loss": train_loss,
            "lr_e": lr_e,
            "lr_d": lr_d,
            "lr_rho": lr_rho,
            **eval_metrics,
        }

        logger.info(
            "[dcam] epoch=%d/%d loss=%.6f T=%d nmi=%s ari=%s acc=%s sc=%s lr_e=%.2e lr_d=%.2e lr_rho=%.2e",
            epoch,
            epochs,
            train_loss,
            curriculum_state.T,
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
                "train/T": curriculum_state.T,
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
                T=curriculum_state.T,
                epoch=epoch,
                total_epochs=epochs,
                output_dir=output_dir,
                logging_cfg=logging_cfg,
                wandb_run=wandb_run,
            )

        if val_rl < best_metric:
            best_metric = val_rl
            best_payload = payload
            _save_best_checkpoint(
                fabric=fabric,
                checkpoint_dir=checkpoint_dir,
                model=model,
                epoch=epoch,
                payload=payload,
                curriculum_state=curriculum_state,
            )

        if float(val_rl) <= float(curriculum_cfg.loss_floor):
            break

    return best_payload
