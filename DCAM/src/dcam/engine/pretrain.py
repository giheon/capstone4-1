# Autoencoder pretraining loop

from __future__ import annotations

from collections.abc import Iterable

import torch
from lightning import Fabric
from tqdm.auto import tqdm



def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - x_hat) ** 2)

def pretrain_autoencoder( # Pretrain e,d by minimizing ||x - d(e(x))||^2
    fabric: Fabric,
    model,
    train_loader: Iterable,
    val_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    logger,
) -> dict[str, float]:
    best_val_rl = float("inf")
    history: dict[str, float] = {}

    for epoch in range(1, epochs + 1):

        model.train()

        train_losses: list[float] = []
        progress = tqdm(train_loader, disable=not fabric.is_global_zero, desc=f"Pretrain {epoch}/{epochs}")

        for batch in progress:
            x = fabric.to_device(batch["x"])
            optimizer.zero_grad(set_to_none=True)

            x_hat = model.reconstruct_without_am(x)
            loss = mse_loss(x=x, x_hat=x_hat)
            fabric.backward(loss)
            optimizer.step()

            train_losses.append(float(loss.detach().item()))
            progress.set_postfix(train_rl=f"{train_losses[-1]:.6f}")

        model.eval()

        val_losses: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch["x"]) == 0:
                    continue
                x = fabric.to_device(batch["x"])
                x_hat = model.reconstruct_without_am(x)
                val_losses.append(float(mse_loss(x=x, x_hat=x_hat).item()))
                
        train_rl = sum(train_losses) / max(len(train_losses), 1)
        val_rl = sum(val_losses) / max(len(val_losses), 1) if val_losses else train_rl
        best_val_rl = min(best_val_rl, val_rl)
        history = {"train_rl": train_rl, "val_rl": val_rl, "best_val_rl": best_val_rl}
        logger.info(f"[pretrain] epoch={epoch} train_rl={train_rl:.6f} val_rl={val_rl:.6f}")

    return history
