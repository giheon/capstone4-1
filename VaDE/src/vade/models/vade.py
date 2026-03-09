"""Main VaDE model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vade.config import TrainConfig
from vade.models.autoencoders.mlp import MLPAutoencoder


EPS = 1e-10


@dataclass(slots=True)
class ForwardOutput:
    x: torch.Tensor
    x_hat: torch.Tensor
    z_mean: torch.Tensor
    z_log_var: torch.Tensor
    z: torch.Tensor
    gamma: torch.Tensor


class VaDEModel(nn.Module):
    """
    Variational Deep Embedding model

    Axis conventions:
        x: [B, D]
        z_mean: [B, J]
        z_log_var: [B, J]
        z: [B, J]
        gamma: [B, K]
        mu_c: [K, J]
        lambda_c: [K, J]
        x_hat: [B, D]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_centroid: int,
        hidden_dims: Sequence[int],
        reconstruction: str,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.n_centroid = int(n_centroid)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.reconstruction = reconstruction

        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.z_mean = nn.Linear(h3, latent_dim)
        self.z_log_var = nn.Linear(h3, latent_dim)

        self.dec1 = nn.Linear(latent_dim, h3)
        self.dec2 = nn.Linear(h3, h2)
        self.dec3 = nn.Linear(h2, h1)
        self.x_bar = nn.Linear(h1, input_dim)

        self.pi_logits = nn.Parameter(torch.zeros(n_centroid))
        self.mu_c = nn.Parameter(torch.zeros(n_centroid, latent_dim))
        self.log_var_c = nn.Parameter(torch.zeros(n_centroid, latent_dim))

        self.reset_parameters()
    
    def reset_parameters(self) -> None: # parameter init
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # The deterministic AE does not provide a variance head.
        # Keep the variance head output near a constant low log-variance at initialization.
        nn.init.zeros_(self.z_log_var.weight)
        nn.init.constant_(self.z_log_var.bias, -4.0)

    def build_autoencoder(self) -> MLPAutoencoder:
        ae = MLPAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            reconstruction=self.reconstruction,
        )
        return ae

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        return h

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: # encoder output
        h = self.encode_hidden(x)
        return self.z_mean(h), self.z_log_var(h)

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        x_hat = self.x_bar(h)
        if self.reconstruction == "sigmoid":
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean=z_mean, z_log_var=z_log_var)
        gamma = self.compute_gamma(z)
        x_hat = self.decode(z)
        return ForwardOutput(
            x=x,
            x_hat=x_hat,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
            gamma=gamma,
        )

    def reconstruct_without_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic AE path used for the paper's pretraining stage."""
        h = self.encode_hidden(x)
        z_det = self.z_mean(h)
        return self.decode(z_det)

    def load_pretrained_autoencoder(self, ae: MLPAutoencoder) -> None:
        with torch.no_grad():
            self.enc1.weight.copy_(ae.enc1.weight)
            self.enc1.bias.copy_(ae.enc1.bias)
            self.enc2.weight.copy_(ae.enc2.weight)
            self.enc2.bias.copy_(ae.enc2.bias)
            self.enc3.weight.copy_(ae.enc3.weight)
            self.enc3.bias.copy_(ae.enc3.bias)
            self.z_mean.weight.copy_(ae.enc4.weight)
            self.z_mean.bias.copy_(ae.enc4.bias)
            self.z_log_var.weight.zero_()
            self.z_log_var.bias.fill_(-4.0)
            self.dec1.weight.copy_(ae.dec1.weight)
            self.dec1.bias.copy_(ae.dec1.bias)
            self.dec2.weight.copy_(ae.dec2.weight)
            self.dec2.bias.copy_(ae.dec2.bias)
            self.dec3.weight.copy_(ae.dec3.weight)
            self.dec3.bias.copy_(ae.dec3.bias)
            self.x_bar.weight.copy_(ae.dec4.weight)
            self.x_bar.bias.copy_(ae.dec4.bias)

    def mixture_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = torch.softmax(self.pi_logits, dim=0)  # theta: [K]
        lambda_c = torch.exp(self.log_var_c)  # lambda_c: [K, J]
        return theta, self.mu_c, lambda_c

    def compute_gamma(self, z: torch.Tensor) -> torch.Tensor:
        theta, mu_c, lambda_c = self.mixture_parameters()

        z_expand = z.unsqueeze(1)  # [B, 1, J]
        mu_expand = mu_c.unsqueeze(0)  # [1, K, J]
        lambda_expand = lambda_c.unsqueeze(0)  # [1, K, J]

        log_theta = torch.log(theta + EPS).view(1, self.n_centroid)  # [1, K]
        log_prob = log_theta
        log_prob = log_prob - 0.5 * torch.sum(torch.log(2.0 * math.pi * lambda_expand + EPS), dim=2)
        log_prob = log_prob - 0.5 * torch.sum(
            (z_expand - mu_expand) ** 2 / (lambda_expand + EPS),
            dim=2,
        )

        log_gamma = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
        return torch.exp(log_gamma)

    def vade_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        
        theta, mu_c, lambda_c = self.mixture_parameters()
        gamma = self.compute_gamma(z)

        if self.reconstruction == "sigmoid":
            recon = F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=1)
        else:
            recon = F.mse_loss(x_hat, x, reduction="none").sum(dim=1)
        recon = alpha * recon

        z_mean_expand = z_mean.unsqueeze(1)  # [B, 1, J]
        z_log_var_expand = z_log_var.unsqueeze(1)  # [B, 1, J]
        mu_expand = mu_c.unsqueeze(0)  # [1, K, J]
        lambda_expand = lambda_c.unsqueeze(0)  # [1, K, J]

        expected_log_p_z_c = 0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(lambda_expand + EPS)
            + torch.exp(z_log_var_expand) / (lambda_expand + EPS)
            + (z_mean_expand - mu_expand) ** 2 / (lambda_expand + EPS)
        ).sum(dim=2)
        kld_like = torch.sum(gamma * expected_log_p_z_c, dim=1)

        z_entropy = -0.5 * torch.sum(1.0 + z_log_var, dim=1)
        cat_prior = -torch.sum(gamma * torch.log(theta.view(1, self.n_centroid) + EPS), dim=1)
        cat_entropy = torch.sum(gamma * torch.log(gamma + EPS), dim=1)

        total = recon + kld_like + z_entropy + cat_prior + cat_entropy
        loss = total.mean()

        gamma_entropy = -(gamma * torch.log(gamma + EPS)).sum(dim=1).mean()
        theta_entropy = -(theta * torch.log(theta + EPS)).sum()
        cluster_usage = torch.bincount(torch.argmax(gamma, dim=1), minlength=self.n_centroid)
        cluster_usage_ratio = cluster_usage.float() / cluster_usage.sum().clamp_min(1).float()
        cluster_usage_entropy = -(cluster_usage_ratio * torch.log(cluster_usage_ratio + EPS)).sum()
        cluster_top1_ratio = cluster_usage_ratio.max()

        logs = {
            "loss": loss.detach(),
            "recon": recon.mean().detach(),
            "kld_like": kld_like.mean().detach(),
            "z_entropy": z_entropy.mean().detach(),
            "cat_prior": cat_prior.mean().detach(),
            "cat_entropy": cat_entropy.mean().detach(),
            "gamma_entropy": gamma_entropy.detach(),
            "theta_entropy": theta_entropy.detach(),
            "cluster_usage_entropy": cluster_usage_entropy.detach(),
            "cluster_top1_ratio": cluster_top1_ratio.detach(),
        }
        return loss, logs

    def nn_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (
            self.enc1,
            self.enc2,
            self.enc3,
            self.z_mean,
            self.z_log_var,
            self.dec1,
            self.dec2,
            self.dec3,
            self.x_bar,
        ):
            params.extend(module.parameters())
        return params

    def gmm_parameters(self) -> list[nn.Parameter]:
        return [self.pi_logits, self.mu_c, self.log_var_c]


def build_model_from_config(config: TrainConfig) -> VaDEModel:
    return VaDEModel(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        n_centroid=config.n_centroid,
        hidden_dims=config.hidden_dims,
        reconstruction=config.reconstruction,
    )
