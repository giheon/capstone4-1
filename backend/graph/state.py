"""
LangGraph State Schema for Math Explanation Generation

Updated Architecture (5 Stages):
1. Node 1: OCR extraction
2. Node 2: Difficulty routing
3. Node 3: Model selection
4. Node 4: Explanation generation (3회 병렬 호출)
5. Node 5: Hard Gate (검증 & 선택)
"""
from typing import TypedDict, Literal, Optional, List, Dict, Any
from dataclasses import dataclass


class MathExplanationState(TypedDict):
    """
    LangGraph State Schema

    LangSmith input/output 형식과 호환되도록 설계
    """

    # ═══════════════════════════════════════════════════════════════
    # 입력 (사용자 제공)
    # ═══════════════════════════════════════════════════════════════
    image_base64: str                                    # 문제 이미지
    explanation_level: Literal["초급", "중급", "고급"]    # 사용자가 선택
    trace_metadata: Dict[str, Any]                       # LangSmith 추적 메타데이터

    # ═══════════════════════════════════════════════════════════════
    # Node 1: OCR 결과
    # ═══════════════════════════════════════════════════════════════
    problem_text: str                                    # 추출된 문제 텍스트
    question_type: Literal["objective", "subjective"]    # 문제 유형 (OCR 판단)
    subject: Literal["확률과통계", "미적분", "기하"]      # 과목
    difficulty: Literal["쉬움", "보통", "어려움", "킬러"]  # 난이도
    unit: str                                            # 단원
    curriculum_area: str                                 # 라우터 입력용 영역
    major_topics: List[str]                              # 라우터 입력용 주제

    # ═══════════════════════════════════════════════════════════════
    # Node 2: 난이도 라우팅 + 모델 선택 결과
    # ═══════════════════════════════════════════════════════════════
    routing_difficulty: Literal["easy", "medium", "hard", "killer"]
    routing_confidence: float
    difficulty_evidence: List[str]
    borderline_with: str
    borderline_reason: str
    selected_model: str                                  # 라우팅된 모델명

    # ═══════════════════════════════════════════════════════════════
    # Node 2: 해설 생성 결과 (3개 후보)
    # ═══════════════════════════════════════════════════════════════
    explanation_candidates: List[dict]
    # 각 후보 형식:
    # {
    #     "concept_explanation": [block, ...],
    #     "problem_review": [block, ...],
    #     "condition_interpretation": [block, ...],
    #     "solution": [block, ...],
    #     "answer": "②" 또는 "17",
    #     "extracted_answer": "②" 또는 "17"
    # }

    # ═══════════════════════════════════════════════════════════════
    # Node 3: Hard Gate 결과
    # ═══════════════════════════════════════════════════════════════
    majority_answer: str                # 다수결로 선정된 답
    majority_candidates: List[dict]     # 다수결 답과 일치하는 후보들
    validation_results: List[dict]      # 검증 결과
    # {
    #     "candidate_index": 0,
    #     "is_valid_json": True/False,
    #     "is_valid_answer_format": True/False,
    #     "is_valid_latex": True/False
    # }
    selected_explanation: dict          # 최종 선택된 해설

    # ═══════════════════════════════════════════════════════════════
    # 최종 출력 (LangSmith output 형식과 동일)
    # ═══════════════════════════════════════════════════════════════
    concept_explanation: List[dict]      # [0. 개념 설명]
    problem_review: List[dict]          # [1. 문제 리뷰]
    condition_interpretation: List[dict] # [2. 조건 해석]
    solution: List[dict]                # [3. 문제 풀이]
    answer: str                         # 최종 답

    # ═══════════════════════════════════════════════════════════════
    # 메타 정보
    # ═══════════════════════════════════════════════════════════════
    stage_timings: Dict[str, float]
    is_complete: bool
    error_message: Optional[str]


@dataclass
class ExplanationCandidate:
    """해설 후보 데이터 클래스"""
    concept_explanation: List[dict]
    problem_review: List[dict]
    condition_interpretation: List[dict]
    solution: List[dict]
    extracted_answer: str
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "concept_explanation": self.concept_explanation,
            "problem_review": self.problem_review,
            "condition_interpretation": self.condition_interpretation,
            "solution": self.solution,
            "extracted_answer": self.extracted_answer,
            "raw_response": self.raw_response,
            "error": self.error
        }


@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    candidate_index: int
    is_valid_json: bool
    is_valid_answer_format: bool
    is_valid_latex: bool

    @property
    def is_all_valid(self) -> bool:
        return self.is_valid_json and self.is_valid_answer_format and self.is_valid_latex

    def to_dict(self) -> dict:
        return {
            "candidate_index": self.candidate_index,
            "is_valid_json": self.is_valid_json,
            "is_valid_answer_format": self.is_valid_answer_format,
            "is_valid_latex": self.is_valid_latex
        }


def create_initial_state(
    image_base64: str,
    explanation_level: str = "중급",
    trace_metadata: Optional[Dict[str, Any]] = None
) -> MathExplanationState:
    """초기 state 생성 헬퍼 함수"""
    return MathExplanationState(
        # 입력
        image_base64=image_base64,
        explanation_level=explanation_level,
        trace_metadata=trace_metadata or {},

        # OCR + 라우팅 결과 (초기화)
        problem_text="",
        question_type="subjective",
        subject="미적분",
        difficulty="보통",
        unit="",
        curriculum_area="",
        major_topics=[],
        routing_difficulty="medium",
        routing_confidence=0.0,
        difficulty_evidence=[],
        borderline_with="none",
        borderline_reason="",
        selected_model="",

        # 해설 생성 결과 (초기화)
        explanation_candidates=[],

        # Hard Gate 결과 (초기화)
        majority_answer="",
        majority_candidates=[],
        validation_results=[],
        selected_explanation={},

        # 최종 출력 (초기화)
        concept_explanation=[],
        problem_review=[],
        condition_interpretation=[],
        solution=[],
        answer="",

        # 메타 정보
        stage_timings={},
        is_complete=False,
        error_message=None
    )
