"""Model routing for subject and routed difficulty."""

from __future__ import annotations

from typing import Dict

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency in local env
    ChatOpenAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None


MODEL_ROUTING: Dict[tuple[str, str], str] = {
    ("기하", "killer"): "gemini-2.5-flash",
    ("기하", "hard"): "gemini-2.5-flash",
    ("기하", "medium"): "gpt-5.4-mini",
    ("기하", "easy"): "gpt-5.4-nano",
    ("미적분", "killer"): "gemini-2.5-pro",
    ("미적분", "hard"): "gpt-5.4",
    ("미적분", "medium"): "gpt-5.4-nano",
    ("미적분", "easy"): "gemini-2.5-flash-lite",
    ("확률과통계", "killer"): "gemini-3-flash-preview",
    ("확률과통계", "hard"): "gpt-5.4",
    ("확률과통계", "medium"): "gpt-5.4-mini",
    ("확률과통계", "easy"): "gemini-2.5-flash-lite",
}

DEFAULT_MODEL = "gpt-5.4"
OCR_MODEL = "gpt-5.4"
DIFFICULTY_ROUTER_MODEL = "gpt-5.4"
EXPLANATION_OVERRIDE_MODEL = "gpt-5.4"

# Runtime aliases keep the public routing map stable while using models that
# are actually available in the current environment.
RUNTIME_MODEL_ALIASES: Dict[str, str] = {
    "gpt-4.5": "gpt-4o",
    "gpt-5-mini": "gpt-5.4-nano",
}


def select_model(subject: str, routed_difficulty: str, explanation_level: str = "중급") -> str:
    """Return the fixed model for a subject / routed difficulty pair.

    초급/고급은 별도의 코칭형 해설 모델로 gpt-4.5를 사용한다.
    """

    if explanation_level in {"초급", "고급"}:
        return EXPLANATION_OVERRIDE_MODEL

    return MODEL_ROUTING.get((subject, routed_difficulty), DEFAULT_MODEL)


def get_model(model_name: str, temperature: float = 0.3):
    """Return a LangChain chat model for OpenAI or Gemini."""
    runtime_model_name = RUNTIME_MODEL_ALIASES.get(model_name, model_name)
    if runtime_model_name.startswith("gemini-"):
        if ChatGoogleGenerativeAI is None:
            raise ImportError(
                "langchain-google-genai is required for Gemini models. "
                "Add it to backend/requirements.txt and install dependencies."
            )
        return ChatGoogleGenerativeAI(model=runtime_model_name, temperature=temperature)

    if ChatOpenAI is None:
        raise ImportError(
            "langchain-openai is required for OpenAI models. "
            "Add it to backend/requirements.txt and install dependencies."
        )

    return ChatOpenAI(model=runtime_model_name, temperature=temperature)
