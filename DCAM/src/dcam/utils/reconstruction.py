"""Shared reconstruction helpers used by pretraining, DCAM training, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_loss_torch(
    x: torch.Tensor,
    decoder_output: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Compute per-sample reconstruction loss and average over the batch."""
    if loss_type == "bce":
        pointwise_loss = F.binary_cross_entropy_with_logits(
            decoder_output,
            x,
            reduction="none",
        )
    else:
        pointwise_loss = (x - decoder_output) ** 2

    reduce_dims = tuple(range(1, pointwise_loss.ndim))
    if not reduce_dims:
        return pointwise_loss.mean()
    return pointwise_loss.sum(dim=reduce_dims).mean()


def reconstruction_loss_numpy(
    x: np.ndarray,
    decoder_output: np.ndarray,
    loss_type: str = "mse",
) -> float:
    """Numpy counterpart to ``reconstruction_loss_torch`` for exported arrays."""
    if loss_type == "bce":
        pointwise_loss = (
            np.maximum(decoder_output, 0.0)
            - (decoder_output * x)
            + np.log1p(np.exp(-np.abs(decoder_output)))
        )
    else:
        pointwise_loss = (x - decoder_output) ** 2

    if pointwise_loss.ndim <= 1:
        return float(np.mean(pointwise_loss))

    per_sample = np.sum(pointwise_loss, axis=tuple(range(1, pointwise_loss.ndim)))
    return float(np.mean(per_sample))
