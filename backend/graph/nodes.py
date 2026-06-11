"""
LangGraph Nodes for Math Explanation Generation

Node 1: OCR extraction - LCEL 사용
Node 2: Difficulty routing - LCEL 사용
Node 3: Model selection - 코드 레벨 고정 라우팅
Node 4: Explanation Generation (3회 병렬 호출) - LCEL 사용
Node 5: Hard Gate (검증 & 선택) - 순수 코드
"""
import json
import re
import random
import asyncio
import time
from typing import Dict, Any, List
from collections import Counter

from .state import MathExplanationState

# Config / prompt / routing imports with dual-path fallback.
try:  # backend package context
    from backend.config import (
        UNITS,
        OCR_TEMPERATURE,
        EXPLANATION_TEMPERATURE,
        PARALLEL_CALL_COUNT,
        DIRECT_EXPLANATION_OUTPUT,
    )
    from backend.prompts import (
        build_ocr_prompt,
        build_difficulty_router_chat_prompt,
        build_explanation_chat_prompt,
    )
    from backend.routing.model_router import (
        DIFFICULTY_ROUTER_MODEL,
        OCR_MODEL,
        get_model,
        select_model,
    )
except ImportError:  # backend directory as working directory
    from config import (
        UNITS,
        OCR_TEMPERATURE,
        EXPLANATION_TEMPERATURE,
        PARALLEL_CALL_COUNT,
        DIRECT_EXPLANATION_OUTPUT,
    )
    from prompts import (
        build_ocr_prompt,
        build_difficulty_router_chat_prompt,
        build_explanation_chat_prompt,
    )
    from routing.model_router import (
        DIFFICULTY_ROUTER_MODEL,
        OCR_MODEL,
        get_model,
        select_model,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 라우팅 보조 함수
# ═══════════════════════════════════════════════════════════════════════════

def normalize_official_difficulty(korean_difficulty: str) -> str:
    mapping = {
        "쉬움": "easy",
        "보통": "medium",
        "어려움": "hard",
        "킬러": "killer",
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
        "killer": "killer",
    }
    return mapping.get(korean_difficulty, "medium")


def derive_curriculum_and_topics(subject: str, unit: str) -> tuple[str, List[str]]:
    if subject == "미적분":
        if unit == "수열":
            return "sequence_limits", ["sequence_limits", "infinite_series"]
        if unit == "미분법":
            return "differentiation", [
                "function_limits",
                "function_continuity",
                "derivative_definition",
                "derivative_applications_basic_1",
                "derivative_applications_basic_2",
                "derivatives_of_various_functions",
                "advanced_differentiation_rules",
                "derivative_applications_advanced",
            ]
        if unit == "적분법":
            return "integration", [
                "indefinite_and_definite_integrals",
                "definite_integral_applications_basic",
                "advanced_integration_methods",
                "definite_integral_applications_advanced",
            ]
        return "differentiation", ["function_limits", "function_continuity"]

    if subject == "확률과통계":
        return "probability_statistics", [unit] if unit else []

    if subject == "기하":
        return "geometry", [unit] if unit else []

    return "unknown", [unit] if unit else []


def _extract_message_text(result: Any) -> str:
    """Coerce LangChain model output into plain text."""
    if isinstance(result, str):
        return result

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)

    return str(result)


def _parse_jsonish_response(result: Any, *, stage: str) -> dict:
    """Parse model output that may include markdown fences or leading text."""
    text = _extract_message_text(result).strip()

    if not text:
        raise ValueError(f"{stage}: empty model output")

    # Remove common markdown code fences first.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parsing first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"{stage}: JSON output is not an object")
    except Exception:
        pass

    # LLMs often emit LaTeX with single backslashes inside JSON strings.
    # Make that shape parseable by escaping bare backslashes once.
    try:
        parsed = json.loads(text.replace("\\", "\\\\"))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: trim to the first JSON object boundaries.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                parsed = json.loads(snippet.replace("\\", "\\\\"))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    raise ValueError(f"{stage}: Invalid json output: {text[:1200]}")


def _stage_timing(state: Dict[str, Any], stage_name: str, started_at: float) -> Dict[str, float]:
    timings = dict(state.get("stage_timings", {}))
    timings[stage_name] = round((time.perf_counter() - started_at) * 1000, 1)
    return timings


# ═══════════════════════════════════════════════════════════════════════════
# Node 1: OCR extraction
# ═══════════════════════════════════════════════════════════════════════════

async def ocr_extraction_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 1: OCR extraction

    - 이미지에서 문제 텍스트 추출
    - 객관식/주관식 판별
    - 과목/난이도/단원 분류
    - 라우터 입력용 curriculum_area / major_topics 생성
    """
    started_at = time.perf_counter()
    image_base64 = state["image_base64"]
    trace_metadata = state.get("trace_metadata", {})

    # LCEL 체인 구성
    prompt = build_ocr_prompt()

    model = get_model(OCR_MODEL, OCR_TEMPERATURE)
    chain = prompt | model

    try:
        raw_result = await chain.ainvoke(
            {"image": image_base64},
            config={
                "run_name": "ocr_extraction",
                "tags": ["ocr", "routing", "vision"],
                "metadata": {
                    **trace_metadata,
                    "node": "ocr_extraction",
                    "model": OCR_MODEL,
                    "explanation_level": state.get("explanation_level", "중급")
                }
            }
        )
        result = _parse_jsonish_response(raw_result, stage="OCR")

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

        curriculum_area, major_topics = derive_curriculum_and_topics(subject, unit)

        return {
            "problem_text": result.get("problem_text", ""),
            "question_type": question_type,
            "subject": subject,
            "difficulty": difficulty,
            "unit": unit,
            "curriculum_area": curriculum_area,
            "major_topics": major_topics,
            "stage_timings": _stage_timing(state, "ocr_extraction", started_at),
        }

    except Exception as e:
        return {
            "problem_text": "",
            "question_type": "subjective",
            "subject": "미적분",
            "difficulty": "보통",
            "unit": "미분법",
            "curriculum_area": "differentiation",
            "major_topics": ["function_limits", "function_continuity"],
            "error_message": f"OCR 오류: {str(e)}"
            ,
            "stage_timings": _stage_timing(state, "ocr_extraction", started_at),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2: Difficulty routing
# ═══════════════════════════════════════════════════════════════════════════

async def difficulty_routing_node(state: MathExplanationState) -> Dict[str, Any]:
    """
    Node 2: Difficulty routing

    - OCR 결과를 바탕으로 routing_difficulty를 판정
    - selected_model은 코드의 고정 매핑으로 선택
    """
    started_at = time.perf_counter()
    subject = state["subject"]
    problem_text = state["problem_text"]
    question_type = state["question_type"]
    difficulty = state["difficulty"]
    unit = state["unit"]
    explanation_level = state.get("explanation_level", "중급")
    curriculum_area = state.get("curriculum_area", "")
    major_topics = state.get("major_topics", [])
    trace_metadata = state.get("trace_metadata", {})

    payload = {
        "curriculum_area": curriculum_area,
        "major_topics": major_topics,
        "problem": problem_text,
        "official_difficulty": normalize_official_difficulty(difficulty),
        "answer_type": question_type,
        "has_image": bool(state.get("image_base64")),
        "subject": subject,
        "unit": unit,
    }

    prompt = build_difficulty_router_chat_prompt(subject)
    model = get_model(DIFFICULTY_ROUTER_MODEL, 0.0)
    chain = prompt | model

    try:
        raw_result = await chain.ainvoke(
            {"payload_json": json.dumps(payload, ensure_ascii=False, indent=2)},
            config={
                "run_name": "difficulty_routing",
                "tags": ["routing", "difficulty", "gpt-4.5"],
                "metadata": {
                    **trace_metadata,
                    "node": "difficulty_routing",
                    "model": DIFFICULTY_ROUTER_MODEL,
                    "subject": subject,
                    "unit": unit,
                }
            }
        )
        result = _parse_jsonish_response(raw_result, stage="DifficultyRouter")

        routing_difficulty = result.get("routing_difficulty", "medium")
        if routing_difficulty not in {"easy", "medium", "hard", "killer"}:
            routing_difficulty = "medium"

        selected_model = select_model(subject, routing_difficulty, explanation_level)

        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        difficulty_evidence = result.get("difficulty_evidence", [])
        if not isinstance(difficulty_evidence, list):
            difficulty_evidence = []

        borderline_with = result.get("borderline_with", "none")
        borderline_reason = result.get("borderline_reason", "")

        return {
            "routing_difficulty": routing_difficulty,
            "routing_confidence": confidence,
            "difficulty_evidence": difficulty_evidence,
            "borderline_with": borderline_with,
            "borderline_reason": borderline_reason,
            "selected_model": selected_model,
            "stage_timings": _stage_timing(state, "difficulty_routing", started_at),
        }

    except Exception as e:
        fallback_difficulty = "medium"
        return {
            "routing_difficulty": fallback_difficulty,
            "routing_confidence": 0.0,
            "difficulty_evidence": [],
            "borderline_with": "none",
            "borderline_reason": "",
            "selected_model": select_model(subject, fallback_difficulty, explanation_level),
            "error_message": f"난이도 라우팅 오류: {str(e)}",
            "stage_timings": _stage_timing(state, "difficulty_routing", started_at),
        }


async def generate_single_explanation(
    problem_text: str,
    subject: str,
    unit: str,
    routed_difficulty: str,
    explanation_level: str,
    question_type: str,
    selected_model: str,
    candidate_index: int = 0,
    trace_metadata: Dict[str, Any] | None = None
) -> dict:
    """단일 해설 생성 (LCEL 사용)"""
    started_at = time.perf_counter()

    prompt = build_explanation_chat_prompt(
        subject=subject,
        difficulty=routed_difficulty,
        explanation_level=explanation_level,
    )

    model = get_model(selected_model, EXPLANATION_TEMPERATURE)
    chain = prompt | model

    try:
        raw_result = await chain.ainvoke(
            {
                "problem_text": problem_text,
                "question_type_label": "객관식" if question_type == "objective" else "주관식",
                "answer_format": "객관식이면 '답: ②', 주관식이면 '답: {숫자}' 형식으로 solution 마지막에 작성하세요.",
            },
            config={
                "run_name": f"generate_explanation_candidate_{candidate_index + 1}",
                "tags": ["explanation-generation", "candidate"],
                "metadata": {
                    **(trace_metadata or {}),
                    "node": "generate_explanation",
                    "candidate_index": candidate_index,
                    "model": selected_model,
                    "subject": subject,
                    "unit": unit,
                    "explanation_level": explanation_level,
                    "question_type": question_type
                }
            }
        )
        result = _parse_jsonish_response(raw_result, stage="ExplanationGeneration")

        # 답 추출
        solution = result.get("solution", "")
        extracted_answer = extract_answer(solution)

        candidate = dict(result)
        candidate.update({
            "problem_review": result.get("problem_review", ""),
            "condition_interpretation": result.get("condition_interpretation", ""),
            "solution": solution,
            "key_points": result.get("key_points", ""),
            "approach_perspectives": result.get("approach_perspectives", ""),
            "transferable_insight": result.get("transferable_insight", ""),
            "extracted_answer": extracted_answer,
            "raw_response": json.dumps(result, ensure_ascii=False),
            "candidate_elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
        })
        return candidate

    except Exception as e:
        return {
            "problem_review": "",
            "condition_interpretation": "",
            "solution": "",
            "key_points": "",
            "approach_perspectives": "",
            "transferable_insight": "",
            "extracted_answer": "",
            "error": str(e),
            "candidate_elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
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
    Node 4: 해설 생성

    - 초급/중급은 같은 프롬프트로 PARALLEL_CALL_COUNT번 병렬 호출
    - 고급은 단발 생성
    - 각 결과에서 답 추출
    """
    started_at = time.perf_counter()
    problem_text = state["problem_text"]
    subject = state["subject"]
    unit = state["unit"]
    explanation_level = state["explanation_level"]
    question_type = state["question_type"]
    selected_model = state["selected_model"]
    routing_difficulty = state["routing_difficulty"]
    trace_metadata = state.get("trace_metadata", {})

    # 고급은 단발 생성, 나머지는 병렬 생성
    candidate_count = 1 if explanation_level == "고급" else PARALLEL_CALL_COUNT
    tasks = [
        generate_single_explanation(
            problem_text,
            subject,
            unit,
            routing_difficulty,
            explanation_level,
            question_type,
            selected_model,
            candidate_index=index,
            trace_metadata=trace_metadata
        )
        for index in range(candidate_count)
    ]

    candidates = await asyncio.gather(*tasks)

    return {
        "explanation_candidates": list(candidates),
        "stage_timings": {
            **dict(state.get("stage_timings", {})),
            "explanation_generation": round((time.perf_counter() - started_at) * 1000, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3: Hard Gate (검증 & 선택)
# ═══════════════════════════════════════════════════════════════════════════

def validate_json_structure(candidate: dict) -> bool:
    """JSON 구조 검증 - 필수 필드 존재 여부"""
    legacy_keys = ["problem_review", "condition_interpretation", "solution"]
    advanced_keys = ["key_points", "approach_perspectives", "transferable_insight"]

    def _has_nonempty_strings(keys: List[str]) -> bool:
        return all(
            key in candidate and
            isinstance(candidate.get(key), str) and
            len(candidate.get(key, "").strip()) > 0
            for key in keys
        )

    return _has_nonempty_strings(legacy_keys) or _has_nonempty_strings(advanced_keys)


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
        candidate.get("solution", "") +
        candidate.get("key_points", "") +
        candidate.get("approach_perspectives", "") +
        candidate.get("transferable_insight", "")
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
    started_at = time.perf_counter()
    candidates = state["explanation_candidates"]
    question_type = state["question_type"]

    # 에러 처리: 후보가 없는 경우
    if not candidates:
        output = _build_error_output("해설 후보가 없습니다.")
        output["stage_timings"] = {
            **dict(state.get("stage_timings", {})),
            "hard_gate": round((time.perf_counter() - started_at) * 1000, 1),
        }
        return output

    # 우회 모드: 후보 1개를 바로 최종 해설로 사용
    if DIRECT_EXPLANATION_OUTPUT:
        selected = candidates[0]
        output = _build_final_output(selected, "", [], [])
        output["stage_timings"] = {
            **dict(state.get("stage_timings", {})),
            "hard_gate": round((time.perf_counter() - started_at) * 1000, 1),
        }
        return output

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: 정답 다수결
    # ─────────────────────────────────────────────────────────────────
    answers = [c.get("extracted_answer", "") for c in candidates]
    answers = [a for a in answers if a]  # 빈 문자열 제거

    if not answers:
        # 답이 하나도 추출 안 됨 → 첫 번째 후보 선택
        selected = candidates[0]
        output = _build_final_output(selected, "", [], [])
        output["stage_timings"] = {
            **dict(state.get("stage_timings", {})),
            "hard_gate": round((time.perf_counter() - started_at) * 1000, 1),
        }
        return output

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

    output = _build_final_output(
        selected,
        majority_answer,
        majority_candidates,
        validation_results
    )
    output["stage_timings"] = {
        **dict(state.get("stage_timings", {})),
        "hard_gate": round((time.perf_counter() - started_at) * 1000, 1),
    }
    return output


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
        "key_points": selected.get("key_points", ""),
        "approach_perspectives": selected.get("approach_perspectives", ""),
        "transferable_insight": selected.get("transferable_insight", ""),
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
        "key_points": "",
        "approach_perspectives": "",
        "transferable_insight": "",
        "answer": "",
        "is_complete": False,
        "error_message": error_message
    }
