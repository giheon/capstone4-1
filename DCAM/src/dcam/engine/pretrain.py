# Autoencoder pretraining loop

from __future__ import annotations

from collections.abc import Iterable

import torch
from lightning import Fabric

from dcam.utils.reconstruction import reconstruction_loss_torch
from dcam.utils.wandb import log_wandb_metrics


def _epoch_average(losses: list[float]) -> float:
    return sum(losses) / max(len(losses), 1)


def pretrain_autoencoder(
    fabric: Fabric,
    model,
    train_loader: Iterable,
    val_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    logger,
    wandb_run=None,
) -> dict[str, float]:
    best_val_loss = float("inf")
    history: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            x = fabric.to_device(batch["x"])
            optimizer.zero_grad(set_to_none=True)
            x_hat_raw = model.reconstruct_without_am(x)
            loss = reconstruction_loss_torch(
                x=x,
                decoder_output=x_hat_raw,
            )
            fabric.backward(loss)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))

        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch["x"]) == 0:
                    continue
                x = fabric.to_device(batch["x"])
                x_hat_raw = model.reconstruct_without_am(x)
                val_losses.append(
                    float(
                        reconstruction_loss_torch(
                            x=x,
                            decoder_output=x_hat_raw,
                        ).item()
                    )
                )

        train_loss = _epoch_average(train_losses)
        val_loss = _epoch_average(val_losses) if val_losses else train_loss
        best_val_loss = min(best_val_loss, val_loss)

        history = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }

        logger.info(
            "[pretrain] epoch=%d/%d loss=%.6f val_loss=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )
        log_wandb_metrics(
            run=wandb_run,
            epoch=epoch,
            axis_name="pretrain/epoch",
            metrics={
                "pretrain/train_loss": train_loss,
                "pretrain/val_loss": val_loss,
            },
        )

    return history
