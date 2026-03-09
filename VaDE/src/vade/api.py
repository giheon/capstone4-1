"""Stable public API and legacy-compatible helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vade.config import TrainConfig, get_default_config, override_config
from vade.data.datasets import load_data
from vade.engine.evaluate import predict_gamma
from vade.engine.train import train_vade
from vade.metrics.clustering import cluster_acc, posterior_numpy, reverse_assignment
from vade.models.vade import VaDEModel, build_model_from_config
from vade.utils.io import save_torch
from vade.utils.seed import seed_everything


VaDE = VaDEModel


def set_seed(seed: int) -> None:
    seed_everything(seed)


def save_checkpoint(
    path: str | Path,
    model: VaDEModel,
    config: TrainConfig,
    history: dict[str, list[Any]],
    seed: int | None = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_kwargs": {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "n_centroid": model.n_centroid,
            "hidden_dims": list(model.hidden_dims),
            "reconstruction": model.reconstruction,
        },
        "train_config": asdict(config),
        "history": history,
        "seed": seed,
    }
    save_torch(path, payload)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[VaDEModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    model = VaDEModel(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload
