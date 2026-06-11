"""OCR and subject/difficulty extraction prompts."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

OCR_ROUTING_SYSTEM_PROMPT = """당신은 수능 수학 문제 분석 전문가입니다.
이미지에서 수학 문제를 정확하게 추출하고 분류하세요."""

OCR_ROUTING_USER_PROMPT = """이미지에서 수학 문제를 분석하세요.

## 작업
1. 문제 텍스트를 정확하게 추출 (수식은 LaTeX 형식)
2. 문제 유형 판별 (객관식/주관식)
3. 과목 분류
4. 난이도 분류
5. 단원 분류

## 과목별 단원
- 확률과통계: 경우의 수, 확률, 확률분포, 통계적 추정
- 미적분: 수열, 미분법, 적분법
- 기하: 이차곡선, 평면벡터, 공간도형과 공간좌표

## 출력 (JSON)
{
    "problem_text": "문제 전체 내용 (LaTeX 수식 포함)",
    "question_type": "objective 또는 subjective",
    "subject": "확률과통계/미적분/기하 중 하나",
    "difficulty": "쉬움/보통/어려움/킬러 중 하나",
    "unit": "해당 과목의 단원명"
}

## 문제 유형 판별 기준
- 보기에 ①②③④⑤가 있으면 → "objective"
- 숫자를 직접 답하는 형식이면 → "subjective"
"""


def _escape_curly_braces(text: str) -> str:
    """Escape literal braces so ChatPromptTemplate does not treat them as variables."""
    return text.replace("{", "{{").replace("}", "}}")


def build_ocr_prompt() -> ChatPromptTemplate:
    """Build the OCR routing prompt as a ChatPromptTemplate."""
    escaped_user_prompt = _escape_curly_braces(OCR_ROUTING_USER_PROMPT)
    return ChatPromptTemplate.from_messages([
        ("system", OCR_ROUTING_SYSTEM_PROMPT),
        ("human", [
            {"type": "text", "text": escaped_user_prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])
