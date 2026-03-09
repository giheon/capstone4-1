"""Autoencoder pretraining for VaDE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vade.config import TrainConfig
from vade.models.autoencoders.mlp import MLPAutoencoder
from vade.models.vade import VaDEModel


BASE_DIR = Path(__file__).resolve().parents[3]


def get_author_pretrain_weight_path(dataset: str) -> Path:
    mapped_dataset = "reuters10k" if dataset == "reuters_all" else dataset
    return BASE_DIR / "pretrain_weights" / f"ae_{mapped_dataset}.pt"


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if hasattr(payload, "state_dict"):
        state = payload.state_dict()
    elif isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        state = payload["model_state"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise TypeError(f"Unsupported pretrain payload type: {type(payload)}")

    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def load_author_pretrained_autoencoder(config: TrainConfig, device: torch.device) -> MLPAutoencoder:
    weight_path = get_author_pretrain_weight_path(config.dataset)
    if not weight_path.exists():
        raise FileNotFoundError(f"author pretrain weights not found: {weight_path}")

    ae = MLPAutoencoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        reconstruction=config.reconstruction,
    ).to(device)

    payload = torch.load(weight_path, map_location=device)
    state = _extract_state_dict(payload)


    expected_keys = set(ae.state_dict())
    state_keys = set(state)

    if expected_keys != state_keys:
        if expected_keys.issubset(state_keys):
            state = {key: state[key] for key in ae.state_dict()}
        else:
            missing = sorted(expected_keys - state_keys)
            extra = sorted(state_keys - expected_keys)
            raise KeyError(
                f"pretrain state dict mismatch for {weight_path}. "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    ae.load_state_dict(state, strict=True)
    return ae


def pretrain_autoencoder(
    features: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    logger=None,
) -> MLPAutoencoder:
    ae = MLPAutoencoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        reconstruction=config.reconstruction,
    ).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=config.pretrain_lr, eps=1e-7)

    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, config.pretrain_epochs + 1):
        ae.train()
        
        running = 0.0
        n = 0

        for batch_x in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_hat = ae(batch_x)
            if config.reconstruction == "sigmoid":
                loss = F.binary_cross_entropy(x_hat, batch_x, reduction="sum") / batch_x.shape[0]
            else:
                loss = F.mse_loss(x_hat, batch_x, reduction="sum") / batch_x.shape[0]
            loss.backward()
            optimizer.step()

            running += float(loss.detach().item()) * batch_x.shape[0]
            n += batch_x.shape[0]

        message = f"[pretrain] epoch={epoch}/{config.pretrain_epochs} loss={running / max(n, 1):.6f}"
        if logger is None:
            print(message)
        else:
            logger.info(message)

    return ae


def initialize_from_pretraining(
    model: VaDEModel,
    features: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    load_pretrained_ae: bool,
    logger=None,
) -> MLPAutoencoder:
    if load_pretrained_ae:
        ae = load_author_pretrained_autoencoder(config=config, device=device)
        message = f"author pretrain weights loaded: {get_author_pretrain_weight_path(config.dataset)}"
    else:
        ae = pretrain_autoencoder(features=features, config=config, device=device, logger=logger)
        message = "autoencoder pretraining finished from scratch"

    model.load_pretrained_autoencoder(ae)
    if logger is None:
        print(message)
    else:
        logger.info(message)
    return ae
