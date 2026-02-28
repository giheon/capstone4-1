#!/usr/bin/env python3
"""Evaluate MNIST VaDE checkpoint and generate class-conditioned samples."""

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vade_core import (
    cluster_acc,
    load_checkpoint,
    load_data,
    posterior_numpy,
    predict_gamma,
    reverse_assignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MNIST VaDE checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/vade_mnist.pt")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--posterior-threshold", type=float, default=0.999)
    parser.add_argument("--max-tries-per-class", type=int, default=100000)
    parser.add_argument("--output-image", default="digits.jpg")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_digit_grid(
    model,
    assignment: np.ndarray,
    device: torch.device,
    samples_per_class: int,
    posterior_threshold: float,
    max_tries_per_class: int,
) -> np.ndarray:
    theta, mu, var = model.mixture_parameters()
    theta = theta.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    var = var.detach().cpu().numpy()

    label_to_cluster = reverse_assignment(assignment)
    grid = np.zeros((28 * 10, 28 * samples_per_class), dtype=np.uint8)

    for label in range(10):
        if label not in label_to_cluster:
            continue
        cluster_idx = label_to_cluster[label]

        count = 0
        tries = 0
        while count < samples_per_class and tries < max_tries_per_class:
            tries += 1
            z = np.random.multivariate_normal(mean=mu[cluster_idx], cov=np.diag(var[cluster_idx]), size=1).astype(np.float32)
            posterior = posterior_numpy(
                z[0],
                theta,
                mu,
                var,
            )
            if posterior[cluster_idx] < posterior_threshold:
                continue

            with torch.no_grad():
                z_t = torch.from_numpy(z).to(device)
                x_hat = model.decode(z_t).cpu().numpy().reshape(28, 28)
            img = np.clip(x_hat * 255.0, 0, 255).astype(np.uint8)
            grid[label * 28 : (label + 1) * 28, count * 28 : (count + 1) * 28] = img
            count += 1

    return grid


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, payload = load_checkpoint(args.checkpoint, device)
    model.eval()

    x, y = load_data("mnist", data_root=args.data_root)
    gamma = predict_gamma(
        model=model,
        features=x,
        batch_size=args.batch_size,
        device=device,
        use_mean=True,
    )

    pred = np.argmax(gamma, axis=1)
    acc, assignment, _ = cluster_acc(pred, y)
    print(f"MNIST dataset VaDE - clustering accuracy: {acc * 100:.2f}%")

    grid = build_digit_grid(
        model=model,
        assignment=assignment,
        device=device,
        samples_per_class=args.samples_per_class,
        posterior_threshold=args.posterior_threshold,
        max_tries_per_class=args.max_tries_per_class,
    )

    out_path = Path(args.output_image)
    Image.fromarray(grid).save(out_path)
    print(f"generated digit grid saved: {out_path}")

    if args.show:
        plt.imshow(grid, cmap=cm.gray)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
