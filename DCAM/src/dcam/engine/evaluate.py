# Evaluation helpers

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import Fabric

from dcam.metrics.clustering import (
    cluster_size_range,
    entropy_of_clusters,
    maybe_supervised_scores,
    reconstruction_loss_numpy,
    silhouette_safe,
)
from dcam.utils.io import save_json, save_numpy
from dcam.utils.visualization import save_image_grid


@torch.no_grad()
def collect_outputs(fabric: Fabric, model, dataloader, T: int):
    model.eval()
    x_all: list[np.ndarray] = []
    x_hat_all: list[np.ndarray] = []
    v_all: list[np.ndarray] = []
    v_prime_all: list[np.ndarray] = []
    c_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    idx_all: list[np.ndarray] = []

    for batch in dataloader:
        x = fabric.to_device(batch["x"])
        y = batch["y"]
        idx = batch["index"]

        v, v_prime, c = model.predict_clusters(x=x, T=T)
        x_hat = model.decode(v_prime)

        x_all.append(fabric.all_gather(x).detach().cpu().numpy())
        x_hat_all.append(fabric.all_gather(x_hat).detach().cpu().numpy())
        v_all.append(fabric.all_gather(v).detach().cpu().numpy())
        v_prime_all.append(fabric.all_gather(v_prime).detach().cpu().numpy())
        c_all.append(fabric.all_gather(c).detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
        idx_all.append(idx.detach().cpu().numpy() if isinstance(idx, torch.Tensor) else np.asarray(idx))

    return {
        "x": np.concatenate(x_all, axis=0),
        "x_hat": np.concatenate(x_hat_all, axis=0),
        "v": np.concatenate(v_all, axis=0),
        "v_prime": np.concatenate(v_prime_all, axis=0),
        "c": np.concatenate(c_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
        "index": np.concatenate(idx_all, axis=0),
    }


@torch.no_grad()
def evaluate_model(fabric: Fabric, model, dataloader, T: int, silhouette_max_samples: int) -> dict[str, float | int | None]:
    payload = collect_outputs(fabric=fabric, model=model, dataloader=dataloader, T=T)
    rl = reconstruction_loss_numpy(payload["x"], payload["x_hat"])
    sc = silhouette_safe(payload["v_prime"], payload["c"], max_samples=silhouette_max_samples)
    etp = entropy_of_clusters(payload["c"])
    cs_max, cs_min = cluster_size_range(payload["c"])
    supervised = maybe_supervised_scores(payload["y"], payload["c"])
    return {
        "rl": rl,
        "sc": sc,
        "entropy": etp,
        "cluster_size_max": cs_max,
        "cluster_size_min": cs_min,
        **supervised,
    }


@torch.no_grad()
def export_inference_artifacts(
    fabric: Fabric,
    model,
    dataloader,
    T: int,
    output_dir: str | Path,
    dataset_type: str,
    num_channels: int | None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = collect_outputs(fabric=fabric, model=model, dataloader=dataloader, T=T)
    if not fabric.is_global_zero:
        return
    rho = model.rho.detach().cpu().numpy()  # rho: [k, m]
    decoded_rho = model.decode_rho().detach().cpu().numpy()  # [k, ...]

    save_numpy(output_dir / "latent_before.npy", payload["v"])
    save_numpy(output_dir / "latent_after.npy", payload["v_prime"])
    save_numpy(output_dir / "rho.npy", rho)
    save_numpy(output_dir / "decoded_prototypes.npy", decoded_rho)

    pd.DataFrame(
        {
            "index": payload["index"].tolist(),
            "cluster": payload["c"].tolist(),
            "label": payload["y"].tolist(),
        }
    ).to_csv(output_dir / "cluster_assignments.csv", index=False)

    metrics = {
        "rl": reconstruction_loss_numpy(payload["x"], payload["x_hat"]),
        "sc": silhouette_safe(payload["v_prime"], payload["c"], max_samples=5000),
    }
    save_json(output_dir / "inference_metrics.json", metrics)

    if dataset_type == "image_folder" and num_channels is not None:
        save_image_grid(decoded_rho, output_dir / "decoded_prototypes.png", num_channels=num_channels)
