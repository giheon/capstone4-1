#!/usr/bin/env python3
"""Evaluate Reuters-all VaDE checkpoint."""

import argparse

import numpy as np
import torch

from vade_core import cluster_acc, load_checkpoint, load_data, predict_gamma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Reuters-all VaDE checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/vade_reuters_all.pt")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    x, y = load_data("reuters_all", data_root=args.data_root)
    gamma = predict_gamma(
        model=model,
        features=x,
        batch_size=args.batch_size,
        device=device,
        use_mean=True,
    )

    pred = np.argmax(gamma, axis=1)
    acc, _, _ = cluster_acc(pred, y)
    print(f"Reuters_all dataset VaDE - clustering accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
