"""CLI entrypoint for VaDE training."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vade.api import (
    build_model_from_config,
    get_default_config,
    load_data,
    override_config,
    predict_gamma,
    save_checkpoint,
    set_seed,
    train_vade,
)
from vade.engine.diagnostics import DiagnosticsConfig
from vade.utils.logging import get_logger
from vade.utils.wandb import finish_wandb, init_wandb, update_wandb_summary


logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VaDE with PyTorch")
    parser.add_argument("dataset", choices=["mnist", "reuters10k", "har", "reuters_all"])
    parser.add_argument("--data-root", default=None, help="Dataset root directory (default: VaDE/dataset)")
    parser.add_argument("--save-path", default=None, help="Checkpoint output path")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--lr-nn", type=float, default=None)
    parser.add_argument("--lr-gmm", type=float, default=None)
    parser.add_argument("--decay-n", type=int, default=None)
    parser.add_argument("--decay-nn", type=float, default=None)
    parser.add_argument("--decay-gmm", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--gmm-reg-covar", type=float, default=None)
    parser.add_argument("--hidden-dims", type=int, nargs=3, default=None)
    parser.add_argument(
        "--pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True: load author pretrain weights from pretrain_weights/*.pt. False: pretrain from scratch.",
    )
    parser.add_argument(
        "--eval-use-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate p(c|z) with z_mean instead of a stochastic sample.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases experiment tracking.",
    )
    parser.add_argument("--wandb-project", default="VaDE", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="W&B team or user entity.")
    parser.add_argument("--wandb-name", default=None, help="W&B run name.")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags for this run.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B run mode when tracking is enabled.",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run GMM and latent-space diagnostics at init, final, and fixed epoch intervals.",
    )
    parser.add_argument(
        "--diagnostics-interval",
        type=int,
        default=50,
        help="Run diagnostics every N epochs during training.",
    )
    parser.add_argument(
        "--tsne-interval",
        type=int,
        default=100,
        help="Run and save t-SNE diagnostics every N epochs during training.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="Directory for diagnostic artifacts. Defaults to <checkpoint>_diagnostics.",
    )
    parser.add_argument(
        "--clean-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove the diagnostics directory before starting the run.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument(
        "--tsne-learning-rate",
        default="auto",
        help='t-SNE learning rate. Use "auto" or a numeric value.',
    )
    parser.add_argument("--tsne-max-iter", type=int, default=1000, help="t-SNE max iterations.")
    parser.add_argument("--tsne-random-state", type=int, default=42, help="t-SNE random seed.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_root = Path(args.data_root) if args.data_root else BASE_DIR / "dataset"
    save_path = Path(args.save_path) if args.save_path else BASE_DIR / "checkpoints" / f"vade_{args.dataset}.pt"
    diagnostics_dir = (
        Path(args.diagnostics_dir)
        if args.diagnostics_dir
        else save_path.parent / f"{save_path.stem}_diagnostics"
    )
    wandb_run = None

    logger.info(f"training on: {args.dataset}")
    logger.info(f"device: {device}")

    base_cfg = get_default_config(args.dataset)
    cfg = override_config(
        base_cfg,
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "lr_nn": args.lr_nn,
            "lr_gmm": args.lr_gmm,
            "decay_n": args.decay_n,
            "decay_nn": args.decay_nn,
            "decay_gmm": args.decay_gmm,
            "alpha": args.alpha,
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_lr": args.pretrain_lr,
            "gmm_reg_covar": args.gmm_reg_covar,
            "hidden_dims": args.hidden_dims,
            "eval_use_mean": args.eval_use_mean,
        },
    )

    x, y = load_data(args.dataset, data_root=data_root)
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    diagnostics_config = DiagnosticsConfig(
        enabled=bool(args.diagnostics),
        interval_epochs=int(args.diagnostics_interval),
        tsne_interval_epochs=int(args.tsne_interval),
        tsne_perplexity=float(args.tsne_perplexity),
        tsne_learning_rate=args.tsne_learning_rate,
        tsne_max_iter=int(args.tsne_max_iter),
        tsne_random_state=int(args.tsne_random_state),
    )
    if diagnostics_config.enabled and diagnostics_config.interval_epochs <= 0:
        raise ValueError("--diagnostics-interval must be a positive integer.")
    if diagnostics_config.enabled and diagnostics_config.tsne_interval_epochs <= 0:
        raise ValueError("--tsne-interval must be a positive integer.")

    if x.shape[1] != cfg.input_dim:
        raise ValueError(f"input dim mismatch: config={cfg.input_dim}, data={x.shape[1]}")

    try:
        if args.clean_diagnostics and diagnostics_dir.exists():
            shutil.rmtree(diagnostics_dir)
            logger.info(f"cleaned diagnostics directory: {diagnostics_dir}")

        wandb_run = init_wandb(
            enabled=bool(args.wandb) and args.wandb_mode != "disabled",
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config={
                **asdict(cfg),
                "device": str(device),
                "load_pretrained_ae": bool(args.pretrain),
                "data_root": str(data_root),
                "diagnostics": asdict(diagnostics_config),
                "diagnostics_dir": str(diagnostics_dir),
                "clean_diagnostics": bool(args.clean_diagnostics),
            },
            mode=args.wandb_mode,
            dir=BASE_DIR / "wandb",
        )

        # Instantiate the VaDE model from the resolved config and move it to the target device.
        model = build_model_from_config(cfg).to(device)
        history = train_vade(
            model=model,
            features=x,
            labels=y,
            config=cfg,
            device=device,
            load_pretrained_ae=args.pretrain,
            eval_use_mean=args.eval_use_mean,
            logger=logger,
            wandb_run=wandb_run,
            diagnostics_config=diagnostics_config,
            diagnostics_dir=diagnostics_dir,
        )

        gamma = predict_gamma(
            model=model,
            features=x,
            batch_size=cfg.batch_size,
            device=device,
            use_mean=args.eval_use_mean,
        )
        pred = np.argmax(gamma, axis=1)
        final_acc = history["acc"][-1] if history["acc"] else 0.0
        logger.info(f"final acc_p_c_z: {final_acc:.6f} | pred shape={pred.shape}")

        save_checkpoint(save_path, model, cfg, history, seed=args.seed)
        logger.info(f"checkpoint saved: {save_path}")

        update_wandb_summary(
            wandb_run,
            {
                "final_acc": float(final_acc),
                "checkpoint_path": str(save_path),
                "diagnostics_dir": str(diagnostics_dir),
                "num_predictions": int(pred.shape[0]),
            },
        )
    except Exception:
        finish_wandb(wandb_run, exit_code=1)
        raise
    else:
        finish_wandb(wandb_run, exit_code=0)


if __name__ == "__main__":
    main()
