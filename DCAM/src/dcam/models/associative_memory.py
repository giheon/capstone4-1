"""Associative memory dynamics used by DCAM.

Mathematical objects:
    rho: [k, m]     learned prototypes / memories
    v:   [B, m]     latent vector before AM
    v':  [B, m]     latent vector after T attractor steps

The paper defines:
    E(v) = -(1 / (2 beta)) log sum_i exp(-beta ||rho_i - v||^2)
    v_{t+1} = v_t - tau * grad_v E(v_t)

Using the closed-form gradient:
    grad_v E(v) = sum_i (rho_i - v) softmax(-beta ||rho_i - v||^2)
"""

from __future__ import annotations

import torch
from torch import nn


class AssociativeMemory(nn.Module):
    def __init__(self, beta: float, tau: float) -> None:
        super().__init__()
        self.beta = beta
        self.tau = tau

    def pairwise_squared_distance(self, v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Return squared distances between batch latents and prototypes.

        Args:
            v: [B, m]
            rho: [k, m]

        Returns:
            dist2: [B, k]
        """
        diff = v[:, None, :] - rho[None, :, :]  # diff: [B, k, m]
        dist2 = (diff * diff).sum(dim=-1)  # dist2: [B, k]
        return dist2

    def energy(self, v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute E(v) for each sample.

        Returns:
            energy: [B]
        """
        dist2 = self.pairwise_squared_distance(v=v, rho=rho)  # [B, k]
        logits = -self.beta * dist2  # [B, k]
        return -(1.0 / (2.0 * self.beta)) * torch.logsumexp(logits, dim=1)  # [B]

    def gradient(self, v: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Closed-form gradient of E(v).

        Args:
            v:   [B, m]
            rho: [k, m]

        Returns:
            grad_E: [B, m] = dE/dv
        """
        dist2 = self.pairwise_squared_distance(v=v, rho=rho)  # [B, k]
        weights = torch.softmax(-self.beta * dist2, dim=1)    # [B, k]

        direction = v[:, None, :] - rho[None, :, :]           # [B, k, m]
        grad_E = (weights[:, :, None] * direction).sum(dim=1) # [B, m]

        return grad_E

    def step(self, v_t: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """One attractor update.

        Args:
            v_t: [B, m]
            rho: [k, m]

        Returns:
            v_{t+1}: [B, m]
        """
        grad_E = self.gradient(v=v_t, rho=rho)  # grad_E: [B, m]
        v_t1 = v_t - self.tau * grad_E  # [B, m]
        return v_t1

    def forward(self, v: torch.Tensor, rho: torch.Tensor, T: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the attractor dynamics operator A^T_rho(v).

        Args:
            v: [B, m]
            rho: [k, m]
            T: number of AM recursion steps

        Returns:
            v_prime: [B, m]
            trace: list with T+1 entries, each [B, m]
        """
        trace = [v]
        v_t = v
        for _ in range(T):
            v_t = self.step(v_t=v_t, rho=rho)
            trace.append(v_t)
        return v_t, trace
