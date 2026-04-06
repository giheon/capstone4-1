from __future__ import annotations

import asyncio
import base64
import json
import re
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
            messages = self._build_messages(
                system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=build_classifier_user_prompt(
                    question_text=question_text,
                    has_image=image_bytes is not None,
                ),
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=16,
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_openai_error(exc)

        raw_text = response.choices[0].message.content.strip()
        # 응답에서 상/중/하 추출
        for difficulty in ["상", "중", "하"]:
            if difficulty in raw_text:
                return difficulty
        return raw_text

    def _generate_solution_sync(
        self,
        model: str,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
        concept_reference: str,
    ) -> str:
        # 모델 매핑 (gpt-5.4 -> gpt-4o)
        model_map = {
            "gpt-5.4": "gpt-4o",
            "gpt-5.4-mini": "gpt-4o-mini",
            "gpt-5.4-nano": "gpt-4o-mini",
        }
        actual_model = model_map.get(model, "gpt-4o")

        try:
            messages = self._build_messages(
                system_prompt=build_solver_system_prompt(concept_reference),
                user_prompt=build_solver_user_prompt(
                    question_text=question_text,
                    has_image=image_bytes is not None,
                ),
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
            response = self._client.chat.completions.create(
                model=actual_model,
                messages=messages,
                max_tokens=2400,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_openai_error(exc)
        return response.choices[0].message.content

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes | None,
        image_mime_type: str | None,
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]
        if image_bytes and image_mime_type:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime_type};base64,{encoded}",
                    },
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _solver_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "extracted_problem": {"type": "string"},
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
            "required": ["extracted_problem", "solution", "answer"],
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
