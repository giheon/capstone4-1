"""Dataset definitions for image folders and vector features."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
from PIL import Image
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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


def _normalize_mnist_pixels(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if float(np.max(x)) > 1.0:
        x = x / 255.0
    return x


def _load_vade_mnist(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = data_root / "mnist" / "mnist.pkl.gz"
    with gzip.open(path, "rb") as handle:
        (x_train, y_train), (x_test, y_test) = pickle.load(handle, encoding="bytes")

    x_train = _normalize_mnist_pixels(x_train).reshape(len(x_train), -1)
    x_test = _normalize_mnist_pixels(x_test).reshape(len(x_test), -1)
    x = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    return x, y


def _load_vade_reuters10k(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = scio.loadmat(data_root / "reuters10k" / "reuters10k.mat")
    x = payload["X"].astype(np.float32)
    y = payload["Y"].squeeze().astype(np.int64)
    return x, y


def _load_vade_har(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = scio.loadmat(data_root / "har" / "HAR.mat")
    x = payload["X"].astype(np.float32)[:10200]
    y = (payload["Y"].squeeze().astype(np.int64) - 1)[:10200]
    return x, y


def _load_vade_reuters_all(
    data_root: Path,
    max_features: int = 2000,
    max_samples: int = 685000,
) -> tuple[np.ndarray, np.ndarray]:
    reuters_root = data_root / "reuters"
    did_to_cat: dict[int, list[str]] = {}
    cat_set = {"CCAT", "GCAT", "MCAT", "ECAT"}

    with (reuters_root / "rcv1-v2.topics.qrels").open("r", encoding="utf-8") as handle:
        for line in handle:
            cat, did_raw, _ = line.strip().split(" ")
            did = int(did_raw)
            if cat in cat_set:
                did_to_cat.setdefault(did, []).append(cat)

    for did in list(did_to_cat):
        if len(did_to_cat[did]) > 1:
            del did_to_cat[did]

    dat_list = [
        "lyrl2004_tokens_test_pt0.dat",
        "lyrl2004_tokens_test_pt1.dat",
        "lyrl2004_tokens_test_pt2.dat",
        "lyrl2004_tokens_test_pt3.dat",
        "lyrl2004_tokens_train.dat",
    ]

    data: list[str] = []
    target: list[int] = []
    cat_to_id = {"CCAT": 0, "GCAT": 1, "MCAT": 2, "ECAT": 3}

    did: int | None = None
    doc = ""
    for dat_name in dat_list:
        with (reuters_root / dat_name).open("r", encoding="latin-1") as handle:
            for line in handle:
                if line.startswith(".I"):
                    if did is not None and doc and did in did_to_cat:
                        data.append(doc)
                        target.append(cat_to_id[did_to_cat[did][0]])
                    did = int(line.strip().split(" ")[1])
                    doc = ""
                    continue
                if line.startswith(".W"):
                    continue
                doc += line

    if did is not None and doc and did in did_to_cat:
        data.append(doc)
        target.append(cat_to_id[did_to_cat[did][0]])

    x = CountVectorizer(dtype=np.float64, max_features=max_features).fit_transform(data)
    x = TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(x)
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    x = preprocessing.normalize(x, norm="l2") * 200.0
    y = np.asarray(target, dtype=np.int64)
    return x[:max_samples].astype(np.float32), y[:max_samples]


def load_vade_benchmark_data(dataset: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    if dataset == "mnist":
        return _load_vade_mnist(root)
    if dataset == "reuters10k":
        return _load_vade_reuters10k(root)
    if dataset == "har":
        return _load_vade_har(root)
    if dataset == "reuters_all":
        return _load_vade_reuters_all(root)
    raise ValueError(f"Unsupported VaDE benchmark dataset: {dataset}")



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
