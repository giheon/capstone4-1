"""MLP autoencoder used for VaDE pretraining."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vade.models.autoencoders.base import BaseAutoencoder


class MLPAutoencoder(BaseAutoencoder):
    """
    MLP autoencoder

    Axis conventions:
        x: [B, D]
        h1: [B, H1]
        h2: [B, H2]
        h3: [B, H3]
        z: [B, J]
        x_hat: [B, D]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        reconstruction: str,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.reconstruction = reconstruction

        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.enc4 = nn.Linear(h3, latent_dim)

        self.dec1 = nn.Linear(latent_dim, h3)
        self.dec2 = nn.Linear(h3, h2)
        self.dec3 = nn.Linear(h2, h1)
        self.dec4 = nn.Linear(h1, input_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc4(self.encode_hidden(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        x_hat = self.dec4(h)
        if self.reconstruction == "sigmoid":
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.enc1, self.enc2, self.enc3, self.enc4):
            params.extend(module.parameters())
        return params

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.dec1, self.dec2, self.dec3, self.dec4):
            params.extend(module.parameters())
        return params
