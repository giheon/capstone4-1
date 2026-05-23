"""
LangSmith Quality Evaluators for Math Explanations

Note: 이 모듈은 오프라인 평가용입니다.
온라인 Hard Gate는 graph/nodes.py의 hard_gate_node에서 코드 레벨로 수행됩니다.

오프라인 평가 기준:
- 조건 사용 완전성 (25%)
- 논리 전개 명확성 (25%)
- 재현 가능성 (30%)
- 수식 표현 정확성 (20%)
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass

from langsmith.evaluation import EvaluationResult, run_evaluator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


@dataclass
class QualityRubric:
    """Quality evaluation rubric (오프라인 평가용)"""
    condition_usage: float = 0.0
    logical_flow: float = 0.0
    reproducibility: float = 0.0
    expression_clarity: float = 0.0  # Changed from latex_accuracy (now no LaTeX)

    @property
    def total_score(self) -> float:
        return (
            self.condition_usage * 0.25 +
            self.logical_flow * 0.25 +
            self.reproducibility * 0.30 +
            self.expression_clarity * 0.20
        )

    @property
    def passes_threshold(self) -> bool:
        return self.total_score >= 0.75


@dataclass
class EvaluationCriteria:
    """Evaluation criteria with weights"""
    condition_usage: float = 0.25      # 조건 사용 완전성
    logical_flow: float = 0.25         # 논리 전개 명확성
    reproducibility: float = 0.30      # 재현 가능성
    expression_clarity: float = 0.20   # 표현 명확성


EVALUATION_PROMPT = """다음 수학 문제 해설을 평가하세요.

## 원본 문제
{problem}

## 해설
### 문제 리뷰
{problem_review}

### 조건 해석
{condition_interpretation}

### 문제 풀이
{solution}

### 답
{answer}

## 평가 기준 (각 0-100점)

1. **조건 사용 완전성 (25%)**
   - 문제의 모든 조건을 활용했는가?
   - 누락된 조건이 없는가?

2. **논리 전개 명확성 (25%)**
   - 각 단계 간 연결이 자연스러운가?
   - 비약 없이 진행되는가?

3. **재현 가능성 (30%)**
   - 학생이 이 해설로 유사문제를 풀 수 있는가?
   - "왜 이렇게 풀었는지"가 명확한가?

4. **표현 명확성 (20%)**
   - 수식이 명확하게 표현되었는가?
   - C(n,r), 1/2 등의 형식이 일관되게 사용되었는가?

## 출력 (JSON)
{{
    "condition_usage": 점수,
    "logical_flow": 점수,
    "reproducibility": 점수,
    "expression_clarity": 점수,
    "feedback": "개선점"
}}
"""


class MathExplanationEvaluator:
    """
    LangSmith compatible evaluator for math explanations.
    오프라인 데이터셋 평가용.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.criteria = EvaluationCriteria()

    async def evaluate(
        self,
        problem: str,
        problem_review: str,
        condition_interpretation: str,
        solution: str,
        answer: str
    ) -> QualityRubric:
        """
        Evaluate a math explanation and return rubric scores.

        Args:
            problem: Original math problem text
            problem_review: 문제 리뷰 섹션
            condition_interpretation: 조건 해석 섹션
            solution: 문제 풀이 섹션
            answer: 최종 답

        Returns:
            QualityRubric with scores
        """
        prompt = EVALUATION_PROMPT.format(
            problem=problem,
            problem_review=problem_review,
            condition_interpretation=condition_interpretation,
            solution=solution,
            answer=answer
        )

        message = HumanMessage(content=prompt)
        response = await self.llm.ainvoke([message])

        try:
            import json
            result = json.loads(response.content)

            rubric = QualityRubric(
                condition_usage=result.get("condition_usage", 70) / 100,
                logical_flow=result.get("logical_flow", 70) / 100,
                reproducibility=result.get("reproducibility", 70) / 100,
                expression_clarity=result.get("expression_clarity", 70) / 100
            )

            return rubric

        except json.JSONDecodeError:
            # Default scores if parsing fails
            return QualityRubric(
                condition_usage=0.7,
                logical_flow=0.7,
                reproducibility=0.7,
                expression_clarity=0.7
            )

    def create_langsmith_evaluator(self):
        """
        Create a LangSmith compatible evaluator function.
        """
        async def evaluate_explanation(
            run,
            example
        ) -> EvaluationResult:
            """LangSmith evaluator function"""
            # Extract inputs and outputs
            problem = example.inputs.get("problem_text", "")

            # New output format
            problem_review = run.outputs.get("problem_review", "")
            condition_interpretation = run.outputs.get("condition_interpretation", "")
            solution = run.outputs.get("solution", "")
            answer = run.outputs.get("answer", "")

            # Run evaluation
            rubric = await self.evaluate(
                problem,
                problem_review,
                condition_interpretation,
                solution,
                answer
            )

            return EvaluationResult(
                key="math_explanation_quality",
                score=rubric.total_score,
                comment=f"조건활용: {rubric.condition_usage:.0%}, "
                        f"논리전개: {rubric.logical_flow:.0%}, "
                        f"재현가능: {rubric.reproducibility:.0%}, "
                        f"표현명확: {rubric.expression_clarity:.0%}"
            )

        return evaluate_explanation


# Create default evaluator instance
default_evaluator = MathExplanationEvaluator()


# Simple evaluators for quick checks
def create_answer_format_evaluator():
    """Evaluator for answer format validation"""
    @run_evaluator
    def answer_format(run, example) -> EvaluationResult:
        answer = run.outputs.get("answer", "")
        question_type = run.outputs.get("question_type", "subjective")

        if question_type == "objective":
            # 객관식: ①②③④⑤ 중 하나
            is_valid = answer in ["①", "②", "③", "④", "⑤"]
        else:
            # 주관식: 숫자
            is_valid = answer.isdigit()

        return EvaluationResult(
            key="answer_format",
            score=1.0 if is_valid else 0.0
        )

    return answer_format


def create_no_latex_evaluator():
    """Evaluator for LaTeX-free output"""
    import re

    @run_evaluator
    def no_latex(run, example) -> EvaluationResult:
        solution = run.outputs.get("solution", "")
        problem_review = run.outputs.get("problem_review", "")
        condition_interpretation = run.outputs.get("condition_interpretation", "")

        full_text = solution + problem_review + condition_interpretation

        latex_patterns = [
            r'\\frac', r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\ln',
            r'\\sqrt', r'\\sum', r'\\int', r'\\lim', r'\\pi',
            r'\$.*\$',
            r'\\[a-zA-Z]+',
        ]

        has_latex = any(re.search(p, full_text) for p in latex_patterns)

        return EvaluationResult(
            key="no_latex",
            score=0.0 if has_latex else 1.0
        )

    return no_latex
