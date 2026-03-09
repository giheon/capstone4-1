"""CLI entrypoint for VaDE evaluation and MNIST sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vade.api import cluster_acc, load_checkpoint, load_data
from vade.engine.evaluate import build_digit_grid, predict_gamma, save_digit_grid
from vade.utils.logging import get_logger


logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VaDE checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default=None)
    common.add_argument("--data-root", default=None)
    common.add_argument("--device", default=None)
    common.add_argument("--batch-size", type=int, default=256)

    mnist = subparsers.add_parser("mnist", parents=[common], help="Evaluate MNIST and sample digits")
    mnist.add_argument("--samples-per-class", type=int, default=10)
    mnist.add_argument("--posterior-threshold", type=float, default=0.999)
    mnist.add_argument("--max-tries-per-class", type=int, default=100000)
    mnist.add_argument("--output-image", default=None)

    subparsers.add_parser("reuters_all", parents=[common], help="Evaluate Reuters-all")
    return parser


def run_mnist(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else BASE_DIR / "checkpoints" / "vade_mnist.pt"
    data_root = Path(args.data_root) if args.data_root else BASE_DIR / "dataset"
    out_path = Path(args.output_image) if args.output_image else BASE_DIR / "digits.jpg"

    model, _ = load_checkpoint(checkpoint_path, device)
    x, y = load_data("mnist", data_root=data_root)
    gamma = predict_gamma(model=model, features=x, batch_size=args.batch_size, device=device, use_mean=True)
    pred = np.argmax(gamma, axis=1)
    acc, assignment, _ = cluster_acc(pred, y)
    logger.info(f"MNIST dataset VaDE - clustering accuracy: {acc * 100:.2f}%")

    grid = build_digit_grid(
        model=model,
        assignment=assignment,
        device=device,
        samples_per_class=args.samples_per_class,
        posterior_threshold=args.posterior_threshold,
        max_tries_per_class=args.max_tries_per_class,
    )
    save_digit_grid(out_path, grid)
    logger.info(f"generated digit grid saved: {out_path}")


def run_reuters_all(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else BASE_DIR / "checkpoints" / "vade_reuters_all.pt"
    )
    data_root = Path(args.data_root) if args.data_root else BASE_DIR / "dataset"

    model, _ = load_checkpoint(checkpoint_path, device)
    x, y = load_data("reuters_all", data_root=data_root)
    gamma = predict_gamma(model=model, features=x, batch_size=args.batch_size, device=device, use_mean=True)
    pred = np.argmax(gamma, axis=1)
    acc, _, _ = cluster_acc(pred, y)
    logger.info(f"Reuters_all dataset VaDE - clustering accuracy: {acc * 100:.2f}%")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "mnist":
        run_mnist(args)
        return
    if args.command == "reuters_all":
        run_reuters_all(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
