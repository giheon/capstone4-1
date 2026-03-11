"""Fully-connected AE used for vector/text data.

Paper-level structure:
    encoder: i -> 500 -> 500 -> 2000 -> d
    decoder: d -> 2000 -> 500 -> 500 -> i

Tensor axis convention:
    x: [B, F]
    v: [B, m]
    x_hat: [B, F]
"""

from __future__ import annotations

from torch import nn

from dcam.models.autoencoders.base import BaseAutoencoder


class MLPAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        reconstruction_loss: str = "mse",
    ) -> None:
        super().__init__(latent_dim=latent_dim, reconstruction_loss=reconstruction_loss)
        h1, h2, h3 = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)  # v: [B, m]

    def decode(self, v):
        return self.decoder(v)  # x_hat raw: [B, F]
