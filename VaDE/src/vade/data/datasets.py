"""Dataset readers for VaDE benchmarks."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as scio
import torch
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parents[3]


def _normalize_mnist_pixels(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    # `mnist.pkl.gz` is often already normalized to [0, 1]. Divide by 255 only
    # when the payload still stores raw 8-bit values.
    if float(np.max(x)) > 1.0:
        x = x / 255.0
    return x


def load_mnist(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = data_root / "mnist" / "mnist.pkl.gz"
    with gzip.open(path, "rb") as handle:
        (x_train, y_train), (x_test, y_test) = pickle.load(handle, encoding="bytes")

    x_train = _normalize_mnist_pixels(x_train).reshape(len(x_train), -1)
    x_test = _normalize_mnist_pixels(x_test).reshape(len(x_test), -1)
    x = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    return x, y


def load_reuters10k(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = scio.loadmat(data_root / "reuters10k" / "reuters10k.mat")
    x = payload["X"].astype(np.float32)
    y = payload["Y"].squeeze().astype(np.int64)
    return x, y


def load_har(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = scio.loadmat(data_root / "har" / "HAR.mat")
    x = payload["X"].astype(np.float32)[:10200]
    y = (payload["Y"].squeeze().astype(np.int64) - 1)[:10200]
    return x, y


def load_reuters_all(
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

    did: Optional[int] = None
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


def load_data(dataset: str, data_root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_root) if data_root is not None else BASE_DIR / "dataset"
    if dataset == "mnist":
        return load_mnist(root)
    if dataset == "reuters10k":
        return load_reuters10k(root)
    if dataset == "har":
        return load_har(root)
    if dataset == "reuters_all":
        return load_reuters_all(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


class ArrayDataset(Dataset):
    """Simple array dataset for vector clustering benchmarks."""

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        y = -1 if self.y is None else self.y[index]
        return {"x": self.x[index], "y": y, "index": index}
