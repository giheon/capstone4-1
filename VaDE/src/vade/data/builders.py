"""DataLoader builders."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from vade.config import get_default_config
from vade.data.datasets import ArrayDataset, load_data
from vade.data.types import DatasetMetadata


def build_dataloaders(
    dataset: str,
    batch_size: int,
    data_root: str | Path | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DatasetMetadata]:
    x, y = load_data(dataset=dataset, data_root=data_root)
    dataset_obj = ArrayDataset(x=x, y=y)
    defaults = get_default_config(dataset)

    persistent_workers = num_workers > 0
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    train_loader = DataLoader(dataset_obj, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(dataset_obj, shuffle=False, **loader_kwargs)

    metadata = DatasetMetadata(
        dataset=dataset,
        input_dim=int(x.shape[1]),
        num_clusters=int(defaults.n_centroid),
        reconstruction=defaults.reconstruction,
        num_samples=len(dataset_obj),
        has_labels=y is not None,
    )
    return train_loader, eval_loader, metadata
