# DCAM fine-tuning loop

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from lightning import Fabric
from tqdm.auto import tqdm

from dcam.engine.evaluate import evaluate_model
from dcam.utils.io import save_json, save_torch



def dcam_reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    r"""Single DCAM loss from the paper.

    For each sample:
        \bar{\ell}(x, e, d, rho) = ||x - d(A^T_rho(e(x)))||^2
    """
    return torch.mean((x - x_hat) ** 2)


@dataclass(slots=True)
class CurriculumState:
    T: int
    lr_reduction_count: int = 0



def _maybe_step_scheduler(scheduler, metric: float) -> bool:
    before = [group["lr"] for group in scheduler.optimizer.param_groups]
    scheduler.step(metric)
    after = [group["lr"] for group in scheduler.optimizer.param_groups]
    return any(a < b for a, b in zip(after, before, strict=True))



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
    scheduler_cfg,
    curriculum_cfg,
    logging_cfg,
    output_dir: str | Path,
    logger,
) -> dict[str, float | int]:
    best_metric = float("inf")
    best_payload: dict[str, float | int] = {}
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):

        model.train()

        train_losses: list[float] = []
        progress = tqdm(train_loader, disable=not fabric.is_global_zero, desc=f"DCAM {epoch}/{epochs}")

        for batch in progress:
            
            x = fabric.to_device(batch["x"])
            opt_e.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            opt_rho.zero_grad(set_to_none=True)

            out = model(x=x, T=curriculum_state.T)
            loss = dcam_reconstruction_loss(x=out.x, x_hat=out.x_hat)
            fabric.backward(loss)
            opt_e.step()
            opt_d.step()
            opt_rho.step()

            loss_value = float(loss.detach().item())
            train_losses.append(loss_value)
            progress.set_postfix(train_loss=f"{loss_value:.6f}", T=curriculum_state.T)

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_metrics = evaluate_model(
            fabric=fabric,
            model=model,
            dataloader=val_loader if len(val_loader) > 0 else eval_loader,
            T=curriculum_state.T,
            silhouette_max_samples=int(logging_cfg.silhouette_max_samples),
        )
        val_rl = float(val_metrics["rl"])

        # scheduler and T curriculum
        reduced_e = _maybe_step_scheduler(sch_e, val_rl)
        reduced_d = _maybe_step_scheduler(sch_d, val_rl)
        reduced_rho = _maybe_step_scheduler(sch_rho, val_rl)
        if reduced_e or reduced_d or reduced_rho:
            curriculum_state.lr_reduction_count += 1
        if curriculum_state.lr_reduction_count >= int(curriculum_cfg.lr_reductions_before_increase_t):
            curriculum_state.T = min(int(curriculum_cfg.t_max), curriculum_state.T + 1)
            curriculum_state.lr_reduction_count = 0

        eval_metrics = evaluate_model(
            fabric=fabric,
            model=model,
            dataloader=eval_loader,
            T=curriculum_state.T,
            silhouette_max_samples=int(logging_cfg.silhouette_max_samples),
        )
        payload = {
            "epoch": epoch,
            "T": curriculum_state.T,
            "train_loss": train_loss,
            **eval_metrics,
        }
        logger.info(
            "[dcam] epoch=%d T=%d train_loss=%.6f rl=%.6f sc=%s nmi=%s ari=%s",
            epoch,
            curriculum_state.T,
            train_loss,
            float(eval_metrics["rl"]),
            f"{eval_metrics['sc']:.4f}" if eval_metrics["sc"] == eval_metrics["sc"] else "nan",
            "nan" if eval_metrics["nmi"] is None else f"{eval_metrics['nmi']:.4f}",
            "nan" if eval_metrics["ari"] is None else f"{eval_metrics['ari']:.4f}",
        )

        if val_rl < best_metric:
            best_metric = val_rl
            best_payload = payload
            if fabric.is_global_zero:
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

        if float(val_rl) <= float(curriculum_cfg.loss_floor):
            break

    return best_payload
