from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from app.core.config import Settings
from app.core.errors import AppError
from app.llm.prompts.classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    build_classifier_user_prompt,
)
from app.llm.prompts.solver import build_solver_system_prompt, build_solver_user_prompt
from app.schemas.response import SolverStructuredOutput


class OpenAIResponsesClient:
    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_seconds,
        )

    async def classify_difficulty(
        self,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
    ) -> str:
        return await asyncio.to_thread(
            self._classify_difficulty_sync,
            question_text,
            image_bytes,
            image_mime_type,
        )

    async def generate_solution(
        self,
        model: str,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
        concept_reference: str,
    ) -> SolverStructuredOutput:
        raw_output = await asyncio.to_thread(
            self._generate_solution_sync,
            model,
            question_text,
            image_bytes,
            image_mime_type,
            concept_reference,
        )
        try:
            parsed = json.loads(raw_output)
            return SolverStructuredOutput.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                error_code="MODEL_OUTPUT_INVALID",
                message="모델이 유효한 JSON 응답을 반환하지 않았습니다.",
                status_code=502,
            ) from exc

    def _classify_difficulty_sync(
        self,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
    ) -> str:
        try:
            response = self._client.responses.create(
                model="gpt-5.4",
                reasoning={"effort": "medium"},
                input=self._build_input(
                    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                    user_prompt=build_classifier_user_prompt(
                        question_text=question_text,
                        has_image=image_bytes is not None,
                    ),
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                ),
                max_output_tokens=16,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_openai_error(exc)
        return response.output_text.strip()

    def _generate_solution_sync(
        self,
        model: str,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
        concept_reference: str,
    ) -> str:
        try:
            response = self._client.responses.create(
                model=model,
                reasoning={"effort": "medium"},
                input=self._build_input(
                    system_prompt=build_solver_system_prompt(concept_reference),
                    user_prompt=build_solver_user_prompt(
                        question_text=question_text,
                        has_image=image_bytes is not None,
                    ),
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                ),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "solver_output",
                        "schema": self._solver_schema(),
                        "strict": True,
                    }
                },
                max_output_tokens=2400,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_openai_error(exc)
        return response.output_text

    @staticmethod
    def _build_input(
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes | None,
        image_mime_type: str | None,
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_prompt},
        ]
        if image_bytes and image_mime_type:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image_mime_type};base64,{encoded}",
                }
            )
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _solver_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "solution": {"type": "string"},
                "answer": {"type": "string"},
                "concept_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "flow": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "order": {"type": "integer"},
                            "concept_id": {"type": "string"},
                        },
                        "required": ["order", "concept_id"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["solution", "answer", "concept_ids", "flow"],
            "additionalProperties": False,
        }

    @staticmethod
    def _raise_openai_error(exc: Exception) -> None:
        if isinstance(exc, AuthenticationError):
            raise AppError(
                error_code="OPENAI_AUTH_ERROR",
                message="OpenAI API 인증에 실패했습니다.",
                status_code=401,
            ) from exc
        if isinstance(exc, RateLimitError):
            raise AppError(
                error_code="OPENAI_RATE_LIMIT",
                message="OpenAI API 요청 제한에 도달했습니다.",
                status_code=429,
            ) from exc
        if isinstance(exc, APITimeoutError):
            raise AppError(
                error_code="OPENAI_TIMEOUT",
                message="OpenAI API 응답 시간이 초과되었습니다.",
                status_code=504,
            ) from exc
        if isinstance(exc, APIConnectionError):
            raise AppError(
                error_code="OPENAI_CONNECTION_ERROR",
                message="OpenAI API 연결에 실패했습니다.",
                status_code=502,
            ) from exc
        if isinstance(exc, APIStatusError):
            raise AppError(
                error_code="OPENAI_UPSTREAM_ERROR",
                message="OpenAI API에서 오류를 반환했습니다.",
                status_code=502,
                details={"upstream_status_code": exc.status_code},
            ) from exc
        raise AppError(
            error_code="OPENAI_UNKNOWN_ERROR",
            message="OpenAI 호출 중 알 수 없는 오류가 발생했습니다.",
            status_code=502,
        ) from exc
