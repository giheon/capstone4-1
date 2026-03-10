"""Visualization helpers for decoded prototypes."""

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def save_image_grid(images: np.ndarray, path: str | Path, num_channels: int) -> None:
    """Save a square grid of decoded prototypes.

    Args:
        images: [K, C, H, W] array for image prototypes.
        path: Output image path.
        num_channels: 1 for grayscale, 3 for RGB.
    """
    k = images.shape[0]
    cols = int(ceil(sqrt(k)))
    rows = int(ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, image in enumerate(images):
        r, c = divmod(idx, cols)
        img = np.transpose(image, (1, 2, 0))
        if num_channels == 1:
            axes[r, c].imshow(img[..., 0], cmap="gray")
        else:
            axes[r, c].imshow(img)
        axes[r, c].set_title(f"ρ[{idx}]", fontsize=10)
        axes[r, c].axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
