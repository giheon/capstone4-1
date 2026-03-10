"""Shared data metadata types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetMetadata:
    """Dataset metadata that later drives model construction.

    Attributes:
        input_shape: For images, [C, H, W]. For vectors, [F].
        num_channels: Number of image channels, or None for vector data.
        num_features: Vector feature dimension, or None for image data.
        has_labels: Whether a label tensor is available.
        dataset_type: 'image_folder' or 'array'.
    """

    input_shape: list[int]
    num_channels: int | None
    num_features: int | None
    has_labels: bool
    dataset_type: str
