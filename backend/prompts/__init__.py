"""Prompt package exports."""

from .ocr import OCR_ROUTING_SYSTEM_PROMPT, OCR_ROUTING_USER_PROMPT, build_ocr_prompt
from .difficulty_router import (
    ROUTING_DIFFICULTY_OUTPUT_SCHEMA,
    ROUTING_DIFFICULTY_PROMPT_STATISTICS,
    ROUTING_DIFFICULTY_PROMPT_CALCULUS,
    ROUTING_DIFFICULTY_PROMPT_GEOMETRY,
    build_difficulty_router_chat_prompt,
)
from .explanation import (
    EXPLANATION_PROMPTS,
    PROMPT_DIFFICULTIES,
    EXPLANATION_LEVELS,
    TOTAL_EXPLANATION_PROMPT_COUNT,
    get_explanation_prompt,
    build_explanation_chat_prompt,
)

__all__ = [
    "OCR_ROUTING_SYSTEM_PROMPT",
    "OCR_ROUTING_USER_PROMPT",
    "build_ocr_prompt",
    "ROUTING_DIFFICULTY_OUTPUT_SCHEMA",
    "ROUTING_DIFFICULTY_PROMPT_STATISTICS",
    "ROUTING_DIFFICULTY_PROMPT_CALCULUS",
    "ROUTING_DIFFICULTY_PROMPT_GEOMETRY",
    "build_difficulty_router_chat_prompt",
    "EXPLANATION_PROMPTS",
    "PROMPT_DIFFICULTIES",
    "EXPLANATION_LEVELS",
    "TOTAL_EXPLANATION_PROMPT_COUNT",
    "get_explanation_prompt",
    "build_explanation_chat_prompt",
]
