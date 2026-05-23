"""
LangGraph Nodes for Math Explanation Generation

Node 1: OCR + Routing (통합) - LCEL 사용
Node 2: Explanation Generation (3회 병렬 호출) - LCEL 사용
Node 3: Hard Gate (검증 & 선택) - 순수 코드
"""
import json
import re
import random
import asyncio
from typing import Dict, Any, List
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import MathExplanationState

# Config에서 설정 import
import sys
sys.path.append('..')
from config import (
    UNITS,
    MODEL_ROUTING,
    DEFAULT_MODEL,
    OCR_MODEL,
    EXPLANATION_PROMPTS,
    FEW_SHOT_EXAMPLES,
    OCR_ROUTING_SYSTEM_PROMPT,
    OCR_ROUTING_USER_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    OCR_TEMPERATURE,
    EXPLANATION_TEMPERATURE,
    PARALLEL_CALL_COUNT
)


# ═══════════════════════════════════════════════════════════════════════════
# 모델 초기화
# ═══════════════════════════════════════════════════════════════════════════

def get_model(model_name: str, temperature: float = 0.3) -> ChatOpenAI:
    """모델 이름으로 ChatOpenAI 인스턴스 생성"""
    return ChatOpenAI(model=model_name, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════════
# Node 1: OCR + 라우팅 (통합)
# ═══════════════════════════════════════════════════════════════════════════

async def ocr_routing_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 1: OCR + 라우팅 (통합)

    - 이미지에서 문제 텍스트 추출
    - 객관식/주관식 판별
    - 과목/난이도/단원 분류
    - 모델 선택 (라우팅)

    LCEL 사용: Prompt | Model | Parser
    """
    image_base64 = state["image_base64"]

    # LCEL 체인 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", OCR_ROUTING_SYSTEM_PROMPT),
        ("human", [
            {"type": "text", "text": OCR_ROUTING_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])

    model = get_model(OCR_MODEL, OCR_TEMPERATURE)
    parser = JsonOutputParser()

    chain = prompt | model | parser

    try:
        result = await chain.ainvoke({"image": image_base64})

        # 결과 추출
        subject = result.get("subject", "미적분")
        difficulty = result.get("difficulty", "보통")
        unit = result.get("unit", "")
        question_type = result.get("question_type", "subjective")

        # 과목 유효성 검사
        if subject not in UNITS:
            subject = "미적분"

        # 단원 유효성 검사
        valid_units = UNITS.get(subject, [])
        if unit not in valid_units and valid_units:
            unit = valid_units[0]

        # 모델 선택 (라우팅)
        selected_model = MODEL_ROUTING.get((subject, difficulty), DEFAULT_MODEL)

        return {
            "problem_text": result.get("problem_text", ""),
            "question_type": question_type,
            "subject": subject,
            "difficulty": difficulty,
            "unit": unit,
            "selected_model": selected_model
        }

    except Exception as e:
        return {
            "problem_text": "",
            "question_type": "subjective",
            "subject": "미적분",
            "difficulty": "보통",
            "unit": "미분법",
            "selected_model": DEFAULT_MODEL,
            "error_message": f"OCR 오류: {str(e)}"
        }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2: 해설 생성 (3회 병렬 호출)
# ═══════════════════════════════════════════════════════════════════════════

def build_explanation_prompt(
    problem_text: str,
    subject: str,
    unit: str,
    explanation_level: str,
    question_type: str
) -> str:
    """
    해설 프롬프트 구성
    - 과목 × 해설수준 프롬프트 선택 (9개 중 1개)
    - Few-shot 예제 주입 (해당 단원 3개)
    """
    # 1. 기본 프롬프트 선택
    base_prompt = EXPLANATION_PROMPTS.get(
        (subject, explanation_level),
        EXPLANATION_PROMPTS.get(("미적분", "중급"), "")
    )

    # 2. Few-shot 예제 가져오기
    examples = FEW_SHOT_EXAMPLES.get(subject, {}).get(unit, [])

    # 3. 예제 포맷팅
    examples_text = ""
    for i, example in enumerate(examples, 1):
        if example and isinstance(example, dict):
            examples_text += f"""
### 예제 {i}
**문제**: {example.get('problem', '')}

**[1. 문제 리뷰]**
{example.get('problem_review', '')}

**[2. 조건 해석]**
{example.get('condition_interpretation', '')}

**[3. 문제 풀이]**
{example.get('solution', '')}
---
"""

    # 4. 답 형식 안내
    answer_format = "객관식이면 '답: ②', 주관식이면 '답: {숫자}' 형식으로 solution 마지막에 작성하세요."

    # 5. 최종 프롬프트 조합
    prompt = f"""
{base_prompt}

## 참고 예제
{examples_text if examples_text else "(예제 없음)"}

## 풀어야 할 문제
{problem_text}

## 문제 유형
{"객관식" if question_type == "objective" else "주관식"}

## 주의사항
- {answer_format}
- 수식은 반드시 LaTeX 형식으로 작성하세요.

위 형식에 맞춰 JSON으로 출력하세요.
"""
    return prompt


async def generate_single_explanation(
    problem_text: str,
    subject: str,
    unit: str,
    explanation_level: str,
    question_type: str,
    selected_model: str
) -> dict:
    """단일 해설 생성 (LCEL 사용)"""

    # 프롬프트 구성
    user_prompt = build_explanation_prompt(
        problem_text, subject, unit, explanation_level, question_type
    )

    # LCEL 체인 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLANATION_SYSTEM_PROMPT),
        ("human", "{user_prompt}")
    ])

    model = get_model(selected_model, EXPLANATION_TEMPERATURE)
    parser = JsonOutputParser()

    chain = prompt | model | parser

    try:
        result = await chain.ainvoke({"user_prompt": user_prompt})

        # 답 추출
        solution = result.get("solution", "")
        extracted_answer = extract_answer(solution)

        return {
            "problem_review": result.get("problem_review", ""),
            "condition_interpretation": result.get("condition_interpretation", ""),
            "solution": solution,
            "extracted_answer": extracted_answer,
            "raw_response": json.dumps(result, ensure_ascii=False)
        }

    except Exception as e:
        return {
            "problem_review": "",
            "condition_interpretation": "",
            "solution": "",
            "extracted_answer": "",
            "error": str(e)
        }


def extract_answer(solution: str) -> str:
    """
    solution 텍스트에서 답 추출
    - 객관식: ①②③④⑤ 중 하나
    - 주관식: 숫자
    """
    if not solution:
        return ""

    # 객관식 패턴: "답: ②" 또는 "답 : ②" 또는 "답:②"
    objective_pattern = r'답\s*:\s*([①②③④⑤])'
    match = re.search(objective_pattern, solution)
    if match:
        return match.group(1)

    # 주관식 패턴: "답: 17" 또는 "답 : 17" 또는 "답:17"
    subjective_pattern = r'답\s*:\s*(\d+)'
    match = re.search(subjective_pattern, solution)
    if match:
        return match.group(1)

    return ""


async def explanation_generation_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 2: 해설 생성 (3회 병렬 호출)

    - 같은 프롬프트로 PARALLEL_CALL_COUNT번 병렬 호출
    - 각 결과에서 답 추출
    """
    problem_text = state["problem_text"]
    subject = state["subject"]
    unit = state["unit"]
    explanation_level = state["explanation_level"]
    question_type = state["question_type"]
    selected_model = state["selected_model"]

    # N회 병렬 호출
    tasks = [
        generate_single_explanation(
            problem_text, subject, unit, explanation_level, question_type, selected_model
        )
        for _ in range(PARALLEL_CALL_COUNT)
    ]

    candidates = await asyncio.gather(*tasks)

    return {
        "explanation_candidates": list(candidates)
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3: Hard Gate (검증 & 선택)
# ═══════════════════════════════════════════════════════════════════════════

def validate_json_structure(candidate: dict) -> bool:
    """JSON 구조 검증 - 필수 필드 존재 여부"""
    required_keys = ["problem_review", "condition_interpretation", "solution"]
    return all(
        key in candidate and
        isinstance(candidate.get(key), str) and
        len(candidate.get(key, "")) > 0
        for key in required_keys
    )


def validate_answer_format(extracted_answer: str, question_type: str) -> bool:
    """
    답 출력 형식 검증
    OCR에서 판단한 question_type과 추출된 답 형식이 일치하는지 확인
    """
    if not extracted_answer:
        return False

    if question_type == "objective":
        # OCR이 객관식이라고 판단 → 답이 ①②③④⑤ 중 하나여야 통과
        return extracted_answer in ["①", "②", "③", "④", "⑤"]
    else:
        # OCR이 주관식이라고 판단 → 답이 숫자여야 통과
        return extracted_answer.isdigit()


def validate_latex(candidate: dict) -> bool:
    """
    LaTeX 문법 검증
    기본적인 LaTeX 패턴이 올바른지 확인
    """
    full_text = (
        candidate.get("problem_review", "") +
        candidate.get("condition_interpretation", "") +
        candidate.get("solution", "")
    )

    # $...$ 패턴 추출
    latex_patterns = re.findall(r'\$[^$]+\$', full_text)

    # 기본적인 검증: 열고 닫는 괄호 매칭
    for pattern in latex_patterns:
        content = pattern[1:-1]  # $ 제거

        # 괄호 매칭 검사
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for char in content:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                expected = brackets[stack.pop()]
                if char != expected:
                    return False

        if stack:  # 닫히지 않은 괄호가 있음
            return False

    return True


async def hard_gate_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 3: Hard Gate (검증 & 선택)

    STEP 1: 정답 다수결 → 2개 이상 일치하는 답 선택
    STEP 2-4: 다수결 후보 내에서만 검증 진행
       - JSON 형식 검증
       - 답 출력 형식 검증 (question_type과 일치)
       - LaTeX 문법 검증
    STEP 5: 최종 선택 (검증 통과 후보 중 랜덤)
    """
    candidates = state["explanation_candidates"]
    question_type = state["question_type"]

    # 에러 처리: 후보가 없는 경우
    if not candidates:
        return _build_error_output("해설 후보가 없습니다.")

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: 정답 다수결
    # ─────────────────────────────────────────────────────────────────
    answers = [c.get("extracted_answer", "") for c in candidates]
    answers = [a for a in answers if a]  # 빈 문자열 제거

    if not answers:
        # 답이 하나도 추출 안 됨 → 첫 번째 후보 선택
        selected = candidates[0]
        return _build_final_output(selected, "", [], [])

    answer_counts = Counter(answers)
    majority_answer, count = answer_counts.most_common(1)[0]

    # 다수결 답과 일치하는 후보들만 필터링
    majority_candidates = [
        c for c in candidates
        if c.get("extracted_answer") == majority_answer
    ]

    # ─────────────────────────────────────────────────────────────────
    # STEP 2-4: 다수결 후보 내에서만 검증 진행
    # ─────────────────────────────────────────────────────────────────
    validation_results = []
    valid_candidates = []

    for i, candidate in enumerate(majority_candidates):
        is_valid_json = validate_json_structure(candidate)
        is_valid_answer = validate_answer_format(
            candidate.get("extracted_answer", ""),
            question_type
        )
        is_valid_latex_result = validate_latex(candidate)

        validation_result = {
            "candidate_index": i,
            "is_valid_json": is_valid_json,
            "is_valid_answer_format": is_valid_answer,
            "is_valid_latex": is_valid_latex_result
        }
        validation_results.append(validation_result)

        # 모든 검증 통과
        if is_valid_json and is_valid_answer and is_valid_latex_result:
            valid_candidates.append(candidate)

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: 최종 선택
    # ─────────────────────────────────────────────────────────────────
    if valid_candidates:
        # 검증 통과한 후보 중 랜덤 선택
        selected = random.choice(valid_candidates)
    elif majority_candidates:
        # 검증 통과한 후보가 없으면 다수결 후보 중 첫 번째
        selected = majority_candidates[0]
    else:
        # 다수결 후보도 없으면 원본 첫 번째
        selected = candidates[0]

    return _build_final_output(
        selected,
        majority_answer,
        majority_candidates,
        validation_results
    )


def _build_final_output(
    selected: dict,
    majority_answer: str,
    majority_candidates: List[dict],
    validation_results: List[dict]
) -> Dict[str, Any]:
    """최종 출력 구성"""
    return {
        "majority_answer": majority_answer,
        "majority_candidates": majority_candidates,
        "validation_results": validation_results,
        "selected_explanation": selected,
        "problem_review": selected.get("problem_review", ""),
        "condition_interpretation": selected.get("condition_interpretation", ""),
        "solution": selected.get("solution", ""),
        "answer": selected.get("extracted_answer", ""),
        "is_complete": True
    }


def _build_error_output(error_message: str) -> Dict[str, Any]:
    """에러 출력 구성"""
    return {
        "majority_answer": "",
        "majority_candidates": [],
        "validation_results": [],
        "selected_explanation": {},
        "problem_review": "",
        "condition_interpretation": "",
        "solution": "",
        "answer": "",
        "is_complete": False,
        "error_message": error_message
    }
