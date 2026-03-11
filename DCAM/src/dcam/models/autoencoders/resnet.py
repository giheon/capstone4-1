"""Residual autoencoder.

The paper specifies a ResNet-inspired AE with filters {32, 64}, repeats=2,
BatchNorm, and LeakyReLU, but does not fully spell out the block-by-block layout.
This module therefore implements a faithful *ResNet-style* AE consistent with those constraints.

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, negative_slope: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


@dataclass(slots=True)
class ConvShapeInfo:
    channels: int
    height: int
    width: int


class ResidualAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_shape: list[int],
        latent_dim: int,
        base_channels: list[int],
        repeats: int,
        negative_slope: float,
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        c, h, w = input_shape
        ch1, ch2 = base_channels

        encoder_layers: list[nn.Module] = [
            nn.Conv2d(c, ch1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]
        in_channels = ch1
        for block_idx in range(repeats):
            encoder_layers.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=ch1,
                    stride=1,
                    negative_slope=negative_slope,
                )
            )
            in_channels = ch1
        encoder_layers.append(
            ResidualBlock(
                in_channels=ch1,
                out_channels=ch2,
                stride=2,
                negative_slope=negative_slope,
            )
        )
        for _ in range(repeats - 1):
            encoder_layers.append(
                ResidualBlock(
                    in_channels=ch2,
                    out_channels=ch2,
                    stride=1,
                    negative_slope=negative_slope,
                )
            )
        encoder_layers.append(nn.Conv2d(ch2, ch2, kernel_size=3, stride=2, padding=1))
        encoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.encoder_conv = nn.Sequential(*encoder_layers)

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
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch2, ch2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(ch1, c, kernel_size=3, stride=1, padding=1),
        )
        self.output_hw = (h, w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)  # h: [B, C', H', W']
        h = h.flatten(start_dim=1)  # h: [B, C'H'W']
        v = self.encoder_fc(h)  # v: [B, m]
        return v

    def decode(self, v: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(v)  # h: [B, C'H'W']
        h = h.view(
            v.shape[0],
            self.shape_info.channels,
            self.shape_info.height,
            self.shape_info.width,
        )
        x_hat = self.decoder(h)
        return x_hat[..., : self.output_hw[0], : self.output_hw[1]]
