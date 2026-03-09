"""Dataset metadata types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetMetadata:
    dataset: str
    input_dim: int
    num_clusters: int
    reconstruction: str
    num_samples: int
    has_labels: bool
