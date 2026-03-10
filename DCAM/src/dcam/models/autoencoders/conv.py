"""Convolutional AE following the DCEC-style layer specification.

Paper-level structure:
    conv5_32 -> conv5_64 -> conv3_128 -> FC_d

Tensor axis convention:
    x: [B, C, H, W]
    v: [B, m]
    x_hat: [B, C, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dcam.models.autoencoders.base import BaseAutoencoder


@dataclass(slots=True)
class ConvShapeInfo:
    channels: int
    height: int
    width: int


class ConvAutoencoder(BaseAutoencoder):
    def __init__(self, input_shape: list[int], latent_dim: int, filters: list[int]) -> None:
        super().__init__(latent_dim=latent_dim)
        c, h, w = input_shape
        f1, f2, f3 = filters

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(c, f1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            encoded = self.encoder_conv(dummy)
        self.shape_info = ConvShapeInfo(
            channels=int(encoded.shape[1]),
            height=int(encoded.shape[2]),
            width=int(encoded.shape[3]),
        )
        flat_dim = int(encoded.numel())

        self.encoder_fc = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(f3, f2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f2, f1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f1, c, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
        self.output_hw = (h, w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)  # h: [B, C3, H3, W3]
        h = h.flatten(start_dim=1)  # h: [B, C3*H3*W3]
        v = self.encoder_fc(h)  # v: [B, m]
        return v

    def decode(self, v: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(v)  # h: [B, C3*H3*W3]
        h = h.view(
            v.shape[0],
            self.shape_info.channels,
            self.shape_info.height,
            self.shape_info.width,
        )  # h: [B, C3, H3, W3]
        x_hat = self.decoder_conv(h)  # x_hat: [B, C, H', W']
        return x_hat[..., : self.output_hw[0], : self.output_hw[1]]
