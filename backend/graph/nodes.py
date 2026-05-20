"""
LangGraph Nodes for Math Explanation Generation
"""
import json
import base64
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import (
    MathExplanationState,
    DIFFICULTY_THRESHOLDS,
    QUALITY_THRESHOLD,
    MAX_RETRIES
)

# Prompts are defined inline to avoid import issues
SYSTEM_PROMPT = """당신은 수능 수학 전문 튜터입니다.
학생이 스스로 문제를 풀 수 있도록 '재현 가능한 사고 과정'을 제시하는 것이 목표입니다.

## 핵심 원칙
1. 모든 조건을 빠짐없이 활용할 것
2. 각 단계에서 "왜 이 방법을 선택했는가"를 명시할 것
3. 수식은 LaTeX 형식으로 작성할 것 (예: $\\frac{1}{2}$, $\\sin\\theta$)
4. 결론 연결 시 "∴" 기호를 사용할 것

## 출력 형식
반드시 아래 JSON 형식으로 출력하세요:

{
    "section_review": "문제 리뷰 내용",
    "section_interpret": "조건 해석 내용",
    "section_solve": "문제 풀이 내용 (STEP 포함)",
    "answer": "최종 정답"
}"""

EXPLANATION_PROMPT = """
## 문제
{problem_text}

## 문제 유형
{problem_type}

## 해설 작성 지침

### 1. 문제 리뷰
- 문제 상황을 간결하게 요약
- 구하고자 하는 것을 명확히 명시

### 2. 조건 해석
- 각 조건의 수학적 의미를 분석
- 조건들 간의 연결고리를 파악

### 3. 문제 풀이
- STEP 단위로 논리적 전개
- 각 STEP에서 사용한 조건과 이유 명시
- 중간 결론은 "∴"로 연결

### 4. 정답
- 최종 답만 간결하게

위 지침에 따라 JSON 형식으로 해설을 작성하세요.
"""

OCR_PROMPT = """이미지에서 수학 문제를 정확하게 텍스트로 추출하세요.
수식은 LaTeX 형식으로 변환하세요.

출력 형식:
{
    "problem_text": "문제 전체 내용",
    "problem_type": "미적분/확률과통계/기하/수학1/수학2"
}"""

DIFFICULTY_PROMPT = """다음 수학 문제의 난이도를 분류하세요.

## 문제
{problem_text}

## 난이도 기준
- 킬러: 복합 개념, 비정형 접근 (수능 21,22,29,30번)
- 준킬러: 2-3개 개념 결합 (수능 20,28번)
- 일반: 단일 개념 (수능 1-19,23-27번)

출력:
{{"difficulty": "킬러/준킬러/일반", "problem_type": "유형", "reasoning": "이유"}}
"""

QUALITY_EVALUATION_PROMPT = """다음 수학 해설의 품질을 평가하세요.

## 원본 문제
{problem_text}

## 생성된 해설
{explanation}

## 평가 기준 (각 0-100점)
1. 조건 사용 완전성 (25%)
2. 논리 전개 명확성 (25%)
3. 재현 가능성 (30%)
4. 수식 표현 정확성 (20%)

출력:
{{"total_score": 점수, "feedback": "개선점", "pass": true/false}}
"""

REGENERATION_PROMPT = """이전 해설이 품질 기준을 충족하지 못했습니다.

## 문제
{problem_text}

## 피드백
{feedback}

## 개선 방향
{improvement_suggestions}

위 피드백을 반영하여 더 나은 해설을 JSON 형식으로 작성하세요.
"""


# Initialize models
gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.3)
gpt4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


async def ocr_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 1: OCR - Extract text from math problem image
    """
    image_base64 = state["image_base64"]

    message = HumanMessage(
        content=[
            {"type": "text", "text": OCR_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            }
        ]
    )

    response = await gpt4o.ainvoke([message])

    try:
        result = json.loads(response.content)
        return {
            "problem_text": result.get("problem_text", ""),
            "problem_type": result.get("problem_type", "수학"),
        }
    except json.JSONDecodeError:
        return {
            "problem_text": response.content,
            "problem_type": "수학",
        }


async def difficulty_classification_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 2: Classify problem difficulty for Hard Gate routing
    """
    problem_text = state["problem_text"]

    prompt = DIFFICULTY_PROMPT.format(problem_text=problem_text)
    message = HumanMessage(content=prompt)

    response = await gpt4o_mini.ainvoke([message])

    try:
        result = json.loads(response.content)
        difficulty = result.get("difficulty", "일반")
        problem_type = result.get("problem_type", state.get("problem_type", "수학"))
    except json.JSONDecodeError:
        difficulty = "일반"
        problem_type = state.get("problem_type", "수학")

    # Hard Gate: Select model based on difficulty
    selected_model = DIFFICULTY_THRESHOLDS.get(difficulty, "gpt-4o-mini")

    return {
        "difficulty": difficulty,
        "problem_type": problem_type,
        "selected_model": selected_model
    }


async def explanation_generation_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 3: Generate explanation using selected model
    """
    problem_text = state["problem_text"]
    problem_type = state["problem_type"]
    selected_model = state["selected_model"]

    # Select model based on Hard Gate decision
    model = gpt4o if selected_model == "gpt-4o" else gpt4o_mini

    prompt = EXPLANATION_PROMPT.format(
        problem_text=problem_text,
        problem_type=problem_type
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = await model.ainvoke(messages)

    try:
        result = json.loads(response.content)
        return {
            "section_review": result.get("section_review", ""),
            "section_interpret": result.get("section_interpret", ""),
            "section_solve": result.get("section_solve", ""),
            "answer": result.get("answer", ""),
        }
    except json.JSONDecodeError:
        # Fallback: parse manually if JSON fails
        content = response.content
        return {
            "section_review": content,
            "section_interpret": "",
            "section_solve": "",
            "answer": "",
        }


async def quality_evaluation_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 4: Soft Quality Gate - Evaluate explanation quality
    """
    problem_text = state["problem_text"]

    explanation = f"""
## 문제 리뷰
{state["section_review"]}

## 조건 해석
{state["section_interpret"]}

## 문제 풀이
{state["section_solve"]}

## 정답
{state["answer"]}
"""

    prompt = QUALITY_EVALUATION_PROMPT.format(
        problem_text=problem_text,
        explanation=explanation
    )

    message = HumanMessage(content=prompt)
    response = await gpt4o_mini.ainvoke([message])

    try:
        result = json.loads(response.content)
        total_score = result.get("total_score", 0) / 100  # Normalize to 0-1
        feedback = result.get("feedback", "")
        passes = result.get("pass", total_score >= QUALITY_THRESHOLD)
    except json.JSONDecodeError:
        total_score = 0.8  # Default pass
        feedback = ""
        passes = True

    return {
        "quality_score": total_score,
        "quality_feedback": feedback,
        "is_complete": passes or state.get("retry_count", 0) >= MAX_RETRIES
    }


async def regeneration_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 5: Regenerate explanation based on quality feedback
    """
    problem_text = state["problem_text"]
    feedback = state["quality_feedback"]
    retry_count = state.get("retry_count", 0)

    prompt = REGENERATION_PROMPT.format(
        problem_text=problem_text,
        feedback=feedback,
        improvement_suggestions=f"품질 점수: {state['quality_score']:.0%}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    # Use GPT-4o for regeneration (higher quality)
    response = await gpt4o.ainvoke(messages)

    try:
        result = json.loads(response.content)
        return {
            "section_review": result.get("section_review", ""),
            "section_interpret": result.get("section_interpret", ""),
            "section_solve": result.get("section_solve", ""),
            "answer": result.get("answer", ""),
            "retry_count": retry_count + 1
        }
    except json.JSONDecodeError:
        return {
            "retry_count": retry_count + 1
        }


def should_regenerate(state: MathExplanationState) -> str:
    """
    Conditional edge: Check if regeneration is needed
    """
    quality_score = state.get("quality_score", 1.0)
    retry_count = state.get("retry_count", 0)

    if quality_score < QUALITY_THRESHOLD and retry_count < MAX_RETRIES:
        return "regenerate"
    return "complete"