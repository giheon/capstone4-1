"""Evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vade.metrics.clustering import cluster_acc, posterior_numpy, reverse_assignment


@torch.no_grad()
def encode_dataset(model, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)

    embeddings: list[np.ndarray] = []
    for batch_x in loader:
        z_mean, _ = model.encode(batch_x.to(device))
        embeddings.append(z_mean.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


@torch.no_grad()
def predict_gamma(
    model,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_mean: bool = True,
) -> np.ndarray:
    model.eval()
    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)

    gamma_all: list[np.ndarray] = []
    for batch_x in loader:
        batch_x = batch_x.to(device)
        z_mean, z_log_var = model.encode(batch_x)
        z = z_mean if use_mean else model.reparameterize(z_mean=z_mean, z_log_var=z_log_var)
        gamma = model.compute_gamma(z)
        gamma_all.append(gamma.cpu().numpy())
    return np.concatenate(gamma_all, axis=0)


@torch.no_grad()
def evaluate_clustering(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_mean: bool = True,
) -> dict[str, np.ndarray | float]:
    gamma = predict_gamma(
        model=model,
        features=features,
        batch_size=batch_size,
        device=device,
        use_mean=use_mean,
    )
    pred = np.argmax(gamma, axis=1)
    acc, assignment, weight = cluster_acc(pred, labels)
    return {"acc": acc, "assignment": assignment, "weight": weight, "gamma": gamma, "pred": pred}


@torch.no_grad()
def build_digit_grid(
    model,
    assignment: np.ndarray,
    device: torch.device,
    samples_per_class: int,
    posterior_threshold: float,
    max_tries_per_class: int,
) -> np.ndarray:
    theta, mu_c, lambda_c = model.mixture_parameters()
    theta_np = theta.detach().cpu().numpy()
    mu_np = mu_c.detach().cpu().numpy()
    var_np = lambda_c.detach().cpu().numpy()

    label_to_cluster = reverse_assignment(assignment)
    grid = np.zeros((28 * 10, 28 * samples_per_class), dtype=np.uint8)

    for label in range(10):
        cluster_idx = label_to_cluster.get(label)
        if cluster_idx is None:
            continue

        count = 0
        tries = 0
        while count < samples_per_class and tries < max_tries_per_class:
            tries += 1
            z = np.random.multivariate_normal(
                mean=mu_np[cluster_idx],
                cov=np.diag(var_np[cluster_idx]),
                size=1,
            ).astype(np.float32)
            posterior = posterior_numpy(z[0], theta_np, mu_np, var_np)
            if posterior[cluster_idx] < posterior_threshold:
                continue

            x_hat = model.decode(torch.from_numpy(z).to(device)).cpu().numpy().reshape(28, 28)
            digit = np.clip(x_hat * 255.0, 0, 255).astype(np.uint8)
            grid[label * 28 : (label + 1) * 28, count * 28 : (count + 1) * 28] = digit
            count += 1

    return grid


def save_digit_grid(path: str | Path, grid: np.ndarray) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
