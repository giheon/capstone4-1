#!/usr/bin/env python3
"""PyTorch VaDE training entrypoint."""

import argparse
from pathlib import Path

import numpy as np
import torch

from vade_core import (
    build_model_from_config,
    get_default_config,
    load_data,
    override_config,
    predict_gamma,
    save_checkpoint,
    set_seed,
    train_vade,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VaDE with PyTorch")
    parser.add_argument("dataset", choices=["mnist", "reuters10k", "har", "reuters_all"])
    parser.add_argument("--data-root", default="dataset", help="Dataset root directory")
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
    parser.add_argument(
        "--pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True: load author pretrain weights from pretrain_weights/*.pt. False: train AE pretrain from scratch.",
    )

    parser.add_argument(
        "--eval-use-mean",
        action="store_true",
        help="Use z_mean for epoch ACC evaluation (default uses stochastic z sampling)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"training on: {args.dataset}")
    print(f"device: {device}")

    base_cfg = get_default_config(args.dataset) # dataset별 기본 하이퍼파라미터 로드
    cfg = override_config( # 사용자가 준 옵션(epochs, batch_size 등)만 base_cfg에 덮어쓴 cfg 생성
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
        },
    )

    x, y = load_data(args.dataset, data_root=args.data_root) # 데이터 가져오기
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    if x.shape[1] != cfg.input_dim:
        raise ValueError(f"input dim mismatch: config={cfg.input_dim}, data={x.shape[1]}")

    model = build_model_from_config(cfg).to(device) # VaDE(nn.Module) 인스턴스 생성 & GPU/CPU에 올림

    history = train_vade(
        model=model,
        features=x,
        labels=y,
        config=cfg,
        device=device,
        load_pretrained_ae=args.pretrain,
        eval_use_mean=args.eval_use_mean,
    )

    gamma = predict_gamma(
        model=model,
        features=x,
        batch_size=cfg.batch_size,
        device=device,
        use_mean=args.eval_use_mean,
    )
    pred = np.argmax(gamma, axis=1)
    final_acc = history["acc"][-1] if history["acc"] else 0.0 # 최종 정확도
    print(f"final acc_p_c_z: {final_acc:.6f} | pred shape={pred.shape}")

    save_path = args.save_path or str(Path("checkpoints") / f"vade_{args.dataset}.pt")
    save_checkpoint(save_path, model, cfg, history)
    print(f"checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
