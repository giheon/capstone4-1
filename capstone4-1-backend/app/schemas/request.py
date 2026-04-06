from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class SolveRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_text: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str | None = None

    @field_validator("question_text", mode="before")
    @classmethod
    def normalize_question_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_presence(self) -> "SolveRequestInput":
        if not self.question_text and not self.image_bytes:
            raise ValueError("question_text 또는 image 중 하나는 필요합니다.")
        return self
