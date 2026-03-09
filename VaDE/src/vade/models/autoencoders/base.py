"""Base autoencoder interface."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class BaseAutoencoder(nn.Module):
    input_dim: int
    latent_dim: int
    reconstruction: str

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError
