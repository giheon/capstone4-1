from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    error_code: str
    message: str
    status_code: int
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"
