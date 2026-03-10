"""Dataset definitions for image folders and vector features."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _slice_samples(
    x: np.ndarray,
    y: np.ndarray | None,
    sample_limit: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if sample_limit is None:
        return x, y
    x = x[:sample_limit]
    y = y[:sample_limit] if y is not None else None
    return x, y


def _prepare_vector_features(x: np.ndarray, normalize: bool) -> np.ndarray:
    x = x.astype(np.float32)
    if normalize:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mean) / std
    return x



class ImageFolderDataset(Dataset):
    """Recursive unlabeled image dataset.

    The dataset scans every image file under ``root``.
    If ``labels_path`` is given, it expects a CSV with columns ``path`` and ``label``.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int,
        num_channels: int,
        normalize: bool,
        mean: list[float],
        std: list[float],
        labels_path: str | Path | None = None,
        sample_limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if sample_limit is not None:
            self.paths = self.paths[:sample_limit]
        if not self.paths:
            raise FileNotFoundError(f"No image files found under: {self.root}")

        transform_ops: list[Callable] = [transforms.Resize((image_size, image_size))]
        if num_channels == 1:
            transform_ops.append(transforms.Grayscale(num_output_channels=1))
        transform_ops.append(transforms.ToTensor())
        if normalize:
            transform_ops.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(transform_ops)

        self.path_to_label: dict[str, int] = {}
        if labels_path is not None:
            frame = pd.read_csv(labels_path)
            self.path_to_label = {
                str((self.root / rel_path).resolve()): int(label)
                for rel_path, label in zip(frame["path"], frame["label"], strict=True)
            }

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        x = self.transform(image)  # x: [C, H, W]
        y = self.path_to_label.get(str(path.resolve()), -1)
        return {"x": x, "y": y, "index": index}


class ArrayDataset(Dataset):
    """Vector dataset for TF-IDF / tabular / precomputed features.

    Supported files:
        - .npy   -> expected shape [N, F]
        - .npz   -> features under ``feature_key`` and optional labels under ``label_key``
        - .csv   -> features in all columns, unless labels_path is provided separately
        - .pt    -> tensor or dict with the same keys as npz
    """

    def __init__(
        self,
        root: str | Path,
        feature_key: str,
        label_key: str,
        labels_path: str | Path | None = None,
        normalize: bool = True,
        sample_limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        x, y = self._load_arrays(self.root, feature_key=feature_key, label_key=label_key)
        if labels_path is not None:
            y = self._load_labels(Path(labels_path))
        x, y = _slice_samples(x=x, y=y, sample_limit=sample_limit)
        x = _prepare_vector_features(x=x, normalize=normalize)
        self.x = torch.from_numpy(x)  # x: [N, F]
        self.y = torch.from_numpy(y.astype(np.int64)) if y is not None else None

    @staticmethod
    def _load_labels(path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            if "label" not in frame.columns:
                raise ValueError("labels CSV must contain a 'label' column")
            return frame["label"].to_numpy()
        raise ValueError(f"Unsupported labels file: {path}")

    @staticmethod
    def _load_arrays(root: Path, feature_key: str, label_key: str) -> tuple[np.ndarray, np.ndarray | None]:
        if root.suffix == ".npy":
            x = np.load(root)
            return x, None
        if root.suffix == ".npz":
            data = np.load(root)
            x = data[feature_key]
            y = data[label_key] if label_key in data.files else None
            return x, y
        if root.suffix == ".csv":
            frame = pd.read_csv(root)
            return frame.to_numpy(), None
        if root.suffix == ".pt":
            payload = torch.load(root, map_location="cpu")
            if isinstance(payload, torch.Tensor):
                return payload.numpy(), None
            x = payload[feature_key]
            y = payload.get(label_key)
            x_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            y_np = y.numpy() if isinstance(y, torch.Tensor) else (None if y is None else np.asarray(y))
            return x_np, y_np
        raise ValueError(f"Unsupported feature file: {root}")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        x = self.x[index]  # x: [F]
        y = -1 if self.y is None else int(self.y[index].item())
        return {"x": x, "y": y, "index": index}


class InMemoryArrayDataset(Dataset):
    """Vector dataset backed by arrays already loaded in memory."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        normalize: bool = False,
        sample_limit: int | None = None,
    ) -> None:
        x, y = _slice_samples(x=x, y=y, sample_limit=sample_limit)
        x = _prepare_vector_features(x=x, normalize=normalize)
        self.x = torch.from_numpy(x)  # x: [N, F]
        self.y = None if y is None else torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        x = self.x[index]  # x: [F]
        y = -1 if self.y is None else int(self.y[index].item())
        return {"x": x, "y": y, "index": index}
