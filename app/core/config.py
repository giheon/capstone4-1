from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    app_env: str
    log_level: str
    openai_api_key: str | None
    max_image_size_mb: int
    request_timeout_seconds: float = 60.0

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    def require_openai_api_key(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_env=os.getenv("APP_ENV", "local"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_image_size_mb=int(os.getenv("MAX_IMAGE_SIZE_MB", "10")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
