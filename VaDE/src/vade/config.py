"""Configuration helpers for VaDE experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    dataset: str
    input_dim: int
    epochs: int
    n_centroid: int
    lr_nn: float
    lr_gmm: float
    decay_n: int
    decay_nn: float
    decay_gmm: float
    alpha: float
    reconstruction: str
    batch_size: int = 100
    latent_dim: int = 10
    hidden_dims: tuple[int, int, int] = (500, 500, 2000)
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    gmm_reg_covar: float = 1e-6
    eval_use_mean: bool = True


def get_default_config(dataset: str) -> TrainConfig:
    if dataset == "mnist":
        return TrainConfig("mnist", 784, 3000, 10, 0.002, 0.002, 10, 0.9, 0.9, 1.0, "sigmoid")
    if dataset == "reuters10k":
        return TrainConfig("reuters10k", 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1.0, "linear")
    if dataset == "har":
        return TrainConfig("har", 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5.0, "linear")
    if dataset == "reuters_all":
        return TrainConfig("reuters_all", 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1.0, "linear")
    raise ValueError(f"Unsupported dataset: {dataset}")


def override_config(config: TrainConfig, args: dict[str, Any]) -> TrainConfig:
    out = copy.deepcopy(config)
    for key, value in args.items():
        if value is None or not hasattr(out, key):
            continue
        if key == "hidden_dims":
            value = tuple(int(dim) for dim in value)
        setattr(out, key, value)
    return out
