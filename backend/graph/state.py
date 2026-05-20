"""
LangGraph State Schema for Math Explanation Generation
"""
from typing import TypedDict, Literal, Optional, List
from dataclasses import dataclass


class MathExplanationState(TypedDict):
    """
    State schema for the math explanation workflow.

    Flow:
    1. image_base64 -> OCR -> problem_text
    2. problem_text -> difficulty classification -> model selection
    3. model -> explanation generation -> sections
    4. sections -> quality evaluation -> final output or retry
    """

    # === Input ===
    image_base64: str

    # === OCR Results ===
    problem_text: str
    problem_type: str  # 미적분, 확률과통계, 기하, 수학1, 수학2
    difficulty: Literal["일반", "준킬러", "킬러"]

    # === Model Selection (Hard Gate) ===
    selected_model: Literal["gpt-4o", "gpt-4o-mini"]

    # === Explanation Sections ===
    # 문제 리뷰: 문제 상황 요약 및 구하고자 하는 것
    section_review: str

    # 조건 해석: 각 조건의 수학적 의미와 연결고리
    section_interpret: str

    # 문제 풀이: STEP별 논리적 전개
    section_solve: str

    # 정답
    answer: str

    # === Quality Evaluation (Soft Gate) ===
    quality_score: float
    quality_feedback: str
    retry_count: int
    max_retries: int

    # === Streaming ===
    current_section: Literal["review", "interpret", "solve", "answer"]
    stream_content: str
    is_formula: bool

    # === Final Output ===
    is_complete: bool
    error_message: Optional[str]


@dataclass
class QualityRubric:
    """
    Quality evaluation rubric based on presentation criteria.
    Total score: 100 points
    """

    # 조건 사용 완전성: 문제에 주어진 모든 조건을 빠짐없이 활용했는가
    condition_usage: float = 0.0  # 25점

    # 논리 전개 명확성: 각 STEP 간 논리적 연결이 명확한가
    logical_flow: float = 0.0  # 25점

    # 재현 가능성: 학생이 이 해설만 보고 유사문제를 풀 수 있는가
    reproducibility: float = 0.0  # 30점

    # 수식 표현 정확성: LaTeX 형식으로 정확하게 작성되었는가
    latex_accuracy: float = 0.0  # 20점

    @property
    def total_score(self) -> float:
        return (
            self.condition_usage * 0.25 +
            self.logical_flow * 0.25 +
            self.reproducibility * 0.30 +
            self.latex_accuracy * 0.20
        )

    @property
    def passes_threshold(self) -> bool:
        """75점 이상이면 통과"""
        return self.total_score >= 0.75


# Difficulty thresholds for Hard Gate routing
DIFFICULTY_THRESHOLDS = {
    "킬러": "gpt-4o",      # 킬러 문제는 GPT-4o 사용
    "준킬러": "gpt-4o",    # 준킬러도 GPT-4o 사용
    "일반": "gpt-4o-mini"  # 일반 문제는 비용 절감
}

# Quality threshold for Soft Gate
QUALITY_THRESHOLD = 0.75  # 75점 미만 시 재생성
MAX_RETRIES = 2  # 최대 재생성 횟수