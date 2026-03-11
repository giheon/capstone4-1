"""Shared squared-error reconstruction helpers used by pretraining, training, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
def reconstruction_loss_torch(
    x: torch.Tensor,
    decoder_output: torch.Tensor,
) -> torch.Tensor:
    """Compute ||x - x_hat||^2 per sample and average over the batch."""
    squared_error = (x - decoder_output) ** 2
    reduce_dims = tuple(range(1, squared_error.ndim))
    if not reduce_dims:
        return squared_error.mean()
    return squared_error.sum(dim=reduce_dims).mean()


def reconstruction_loss_numpy(
    x: np.ndarray,
    decoder_output: np.ndarray,
) -> float:
    """Numpy counterpart to ``reconstruction_loss_torch``."""
    squared_error = (x - decoder_output) ** 2
    if squared_error.ndim <= 1:
        return float(np.mean(squared_error))
    per_sample = np.sum(squared_error, axis=tuple(range(1, squared_error.ndim)))
    return float(np.mean(per_sample))
