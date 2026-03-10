"""Base autoencoder interfaces."""

from __future__ import annotations

import torch
from torch import nn


class BaseAutoencoder(nn.Module):
    """Minimal interface every DCAM backbone should follow."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encoder_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith('encoder')]

    def decoder_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('encoder')]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
