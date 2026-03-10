"""Main DCAM model.

The model composes:
    x --e--> v --A^T_rho--> v' --d--> x_hat

Axis conventions:
    image x: [B, C, H, W]
    vector x: [B, F]
    v: [B, m]
    rho: [k, m]
    x_hat: same shape as x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from dcam.data.types import DatasetMetadata

from dcam.models.associative_memory import AssociativeMemory
from dcam.models.autoencoders.base import BaseAutoencoder
from dcam.models.autoencoders.conv import ConvAutoencoder
from dcam.models.autoencoders.mlp import MLPAutoencoder
from dcam.models.autoencoders.resnet import ResidualAutoencoder


@dataclass(slots=True)
class ForwardOutput:
    x: torch.Tensor
    v: torch.Tensor
    v_prime: torch.Tensor
    x_hat: torch.Tensor
    trace: list[torch.Tensor]


class DCAMModel(nn.Module):
    def __init__(self, ae: BaseAutoencoder, num_clusters: int, beta: float, tau: float) -> None:
        super().__init__()
        self.ae = ae
        self.m = ae.latent_dim
        self.k = num_clusters

        # rho: [k, m] learned cluster prototypes in latent space.
        self.rho = nn.Parameter(torch.empty(num_clusters, self.m))
        nn.init.normal_(self.rho, mean=0.0, std=0.02)

        self.am = AssociativeMemory(beta=beta, tau=tau)

    @property
    def e(self) -> BaseAutoencoder:
        return self.ae

    @property
    def d(self) -> BaseAutoencoder:
        return self.ae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(x)  # v: [B, m]

    def decode(self, v: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(v)  # x_hat: same shape as x

    def forward(self, x: torch.Tensor, T: int) -> ForwardOutput:
        v = self.encode(x)  # v: [B, m]
        v_prime, trace = self.am(v=v, rho=self.rho, T=T)  # v_prime: [B, m]
        x_hat = self.decode(v_prime)  # x_hat: same shape as x
        return ForwardOutput(x=x, v=v, v_prime=v_prime, x_hat=x_hat, trace=trace)

    def reconstruct_without_am(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def initialize_rho_from_batch(self, x: torch.Tensor) -> None:
        """Initialize rho from encoded random samples, matching Algorithm 1.

        Args:
            x: [B, ...]
        """
        v = self.encode(x)  # [B, m]
        if v.shape[0] < self.k:
            raise ValueError(
                f"Need at least k={self.k} samples in the init batch, got batch_size={v.shape[0]}"
            )
        perm = torch.randperm(v.shape[0], device=v.device)[: self.k]
        self.rho.data.copy_(v[perm].detach())

    @torch.no_grad()
    def predict_clusters(self, x: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return post-AM latent vectors and hard cluster IDs.

        Returns:
            v: [B, m]
            v_prime: [B, m]
            c: [B]
        """
        v = self.encode(x)
        v_prime, _ = self.am(v=v, rho=self.rho, T=T)
        dist2 = self.am.pairwise_squared_distance(v=v_prime, rho=self.rho)  # [B, k]
        c = dist2.argmin(dim=1)  # [B]
        return v, v_prime, c

    @torch.no_grad()
    def decode_rho(self) -> torch.Tensor:
        """Decode learned prototypes back into input space.

        Returns:
            decoded prototypes: [k, ...]
        """
        return self.decode(self.rho)



def build_autoencoder(model_cfg: DictConfig, metadata: DatasetMetadata) -> BaseAutoencoder:
    backbone = model_cfg.backbone
    latent_dim = int(model_cfg.latent_dim)
    if backbone == "cae":
        return ConvAutoencoder(
            input_shape=list(model_cfg.input_shape or metadata.input_shape),
            latent_dim=latent_dim,
            filters=list(model_cfg.cae.filters),
        )
    if backbone == "rae":
        return ResidualAutoencoder(
            input_shape=list(model_cfg.input_shape or metadata.input_shape),
            latent_dim=latent_dim,
            base_channels=list(model_cfg.rae.base_channels),
            repeats=int(model_cfg.rae.repeats),
            negative_slope=float(model_cfg.rae.negative_slope),
        )
    if backbone == "eae":
        input_dim = int(metadata.num_features or metadata.input_shape[0])
        return MLPAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=list(model_cfg.eae.hidden_dims),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")



def build_dcam_model(cfg: DictConfig, metadata: DatasetMetadata) -> DCAMModel:
    ae = build_autoencoder(model_cfg=cfg.model, metadata=metadata)
    return DCAMModel(
        ae=ae,
        num_clusters=int(cfg.model.num_clusters),
        beta=float(cfg.model.beta),
        tau=float(cfg.model.tau),
    )
