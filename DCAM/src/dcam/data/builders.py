"""Data loader builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split

from dcam.data.datasets import (
    ArrayDataset,
    ImageFolderDataset,
    InMemoryArrayDataset,
    load_vade_benchmark_data,
)
from dcam.data.types import DatasetMetadata


def _resolve_vade_data_root(data_root: str | None) -> str | None:
    if data_root is not None:
        return data_root
    default_root = Path(__file__).resolve().parents[3] / "data" / "raw" / "vade"
    if default_root.exists():
        return str(default_root)
    return None



def build_datasets(cfg: DictConfig) -> tuple[Any, Any, Any, DatasetMetadata]:
    """Build train/val/eval datasets from Hydra config."""
    dataset_type = cfg.dataset_type
    if dataset_type == "image_folder":
        full_dataset = ImageFolderDataset(
            root=cfg.root,
            image_size=cfg.image_size,
            num_channels=cfg.num_channels,
            normalize=cfg.normalize,
            mean=list(cfg.mean),
            std=list(cfg.std),
            labels_path=cfg.labels_path,
            sample_limit=cfg.sample_limit,
        )
        metadata = DatasetMetadata(
            input_shape=[cfg.num_channels, cfg.image_size, cfg.image_size],
            num_channels=int(cfg.num_channels),
            num_features=None,
            has_labels=cfg.labels_path is not None,
            dataset_type=dataset_type,
        )
    elif dataset_type == "array":
        full_dataset = ArrayDataset(
            root=cfg.root,
            feature_key=cfg.feature_key,
            label_key=cfg.label_key,
            labels_path=cfg.labels_path,
            normalize=cfg.normalize,
            sample_limit=cfg.sample_limit,
        )
        metadata = DatasetMetadata(
            input_shape=[int(full_dataset.x.shape[1])],
            num_channels=None,
            num_features=int(full_dataset.x.shape[1]),
            has_labels=full_dataset.y is not None,
            dataset_type=dataset_type,
        )
    elif dataset_type == "vade_benchmark":
        data_root = _resolve_vade_data_root(cfg.data_root)
        if data_root is None:
            raise FileNotFoundError(
                "VaDE benchmark data root was not found. Copy the dataset under "
                "`DCAM/data/raw/vade` or set `data.data_root=/path/to/dataset`."
            )
        x, y = load_vade_benchmark_data(
            dataset=str(cfg.dataset_name),
            data_root=data_root,
        )
        full_dataset = InMemoryArrayDataset(
            x=x,
            y=y,
            normalize=bool(cfg.normalize),
            sample_limit=cfg.sample_limit,
        )
        metadata = DatasetMetadata(
            input_shape=[int(full_dataset.x.shape[1])],
            num_channels=None,
            num_features=int(full_dataset.x.shape[1]),
            has_labels=full_dataset.y is not None,
            dataset_type=dataset_type,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    val_size = int(round(len(full_dataset) * float(cfg.val_fraction)))
    val_size = min(max(val_size, 1), len(full_dataset) - 1) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = full_dataset
        val_dataset = Subset(full_dataset, [])

    eval_dataset = full_dataset
    return train_dataset, val_dataset, eval_dataset, metadata



def build_dataloaders(cfg: DictConfig):
    train_dataset, val_dataset, eval_dataset, metadata = build_datasets(cfg)

    persistent_workers = bool(cfg.persistent_workers) and int(cfg.num_workers) > 0
    common_loader_kwargs = dict(
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_loader_kwargs)
    eval_loader = DataLoader(eval_dataset, shuffle=False, drop_last=False, **common_loader_kwargs)
    return train_loader, val_loader, eval_loader, metadata
