"""Main DCAM model.

The model composes:
    x --e--> v --A^T_rho--> v' --d--> x_hat

Axis conventions:
    image x: [B, C, H, W]
    vector x: [B, F]
    v: [B, m]
    rho: [k, m]
    x_hat_raw: raw decoder output, same shape as x
    x_hat: projected observation used by diagnostics/export
"""

from __future__ import annotations

from dataclasses import dataclass

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
    x_hat_raw: torch.Tensor
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(x)  # v: [B, m]

    def decode(self, v: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(v)  # raw decoder output: same shape as x

    def decode_outputs(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw decoder output and the projected observation used for visualization."""
        x_hat_raw = self.decode(v)
        x_hat = self.ae.project_reconstruction(x_hat_raw)
        return x_hat_raw, x_hat

    def decode_observation(self, v: torch.Tensor) -> torch.Tensor:
        return self.decode_outputs(v)[1]

    def forward(self, x: torch.Tensor, T: int) -> ForwardOutput:
        v = self.encode(x)  # v: [B, m]
        v_prime, trace = self.am(v=v, rho=self.rho, T=T)  # v_prime: [B, m]
        x_hat_raw, x_hat = self.decode_outputs(v_prime)
        return ForwardOutput(x=x, v=v, v_prime=v_prime, x_hat_raw=x_hat_raw, x_hat=x_hat, trace=trace)
    
    def reconstruct_without_am(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw decoder output used by the paper's squared-error loss."""
        return self.decode(self.encode(x))

    @torch.no_grad()
    def initialize_rho_from_centroids(self, centroids: torch.Tensor) -> None:
        """Initialize rho from precomputed latent centroids.

        Args:
            centroids: [k, m]
        """
        if centroids.shape != self.rho.shape:
            raise ValueError(
                f"Expected centroid shape {tuple(self.rho.shape)}, got {tuple(centroids.shape)}"
            )
        self.rho.data.copy_(centroids.detach())

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
        return self.decode_observation(self.rho)



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
