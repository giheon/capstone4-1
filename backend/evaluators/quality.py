"""
LangSmith Quality Evaluators for Math Explanations
Based on rubric from presentation script.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass

from langsmith.evaluation import EvaluationResult, run_evaluator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dataclasses import dataclass as dc


@dc
class QualityRubric:
    """Quality evaluation rubric"""
    condition_usage: float = 0.0
    logical_flow: float = 0.0
    reproducibility: float = 0.0
    latex_accuracy: float = 0.0

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
        return self.total_score >= 0.75


@dataclass
class EvaluationCriteria:
    """Evaluation criteria with weights"""
    condition_usage: float = 0.25      # 조건 사용 완전성
    logical_flow: float = 0.25         # 논리 전개 명확성
    reproducibility: float = 0.30      # 재현 가능성
    latex_accuracy: float = 0.20       # 수식 표현 정확성


EVALUATION_PROMPT = """다음 수학 문제 해설을 평가하세요.

## 원본 문제
{problem}

## 해설
{explanation}

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

4. **수식 표현 정확성 (20%)**
   - LaTeX 문법이 올바른가?
   - 수학적 표기가 정확한가?

## 출력 (JSON)
{{
    "condition_usage": 점수,
    "logical_flow": 점수,
    "reproducibility": 점수,
    "latex_accuracy": 점수,
    "feedback": "개선점"
}}
"""


class MathExplanationEvaluator:
    """
    LangSmith compatible evaluator for math explanations.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.criteria = EvaluationCriteria()

    async def evaluate(
        self,
        problem: str,
        explanation: str
    ) -> QualityRubric:
        """
        Evaluate a math explanation and return rubric scores.

        Args:
            problem: Original math problem text
            explanation: Generated explanation text

        Returns:
            QualityRubric with scores
        """
        prompt = EVALUATION_PROMPT.format(
            problem=problem,
            explanation=explanation
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
                latex_accuracy=result.get("latex_accuracy", 70) / 100
            )

            return rubric

        except json.JSONDecodeError:
            # Default scores if parsing fails
            return QualityRubric(
                condition_usage=0.7,
                logical_flow=0.7,
                reproducibility=0.7,
                latex_accuracy=0.7
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
            explanation = run.outputs.get("explanation", "")

            # Run evaluation
            rubric = await self.evaluate(problem, explanation)

            return EvaluationResult(
                key="math_explanation_quality",
                score=rubric.total_score,
                comment=f"조건활용: {rubric.condition_usage:.0%}, "
                        f"논리전개: {rubric.logical_flow:.0%}, "
                        f"재현가능: {rubric.reproducibility:.0%}, "
                        f"수식정확: {rubric.latex_accuracy:.0%}"
            )

        return evaluate_explanation


# Individual criterion evaluators for LangSmith
def create_condition_usage_evaluator():
    """Evaluator for condition usage completeness"""
    @run_evaluator
    def condition_usage(run, example) -> EvaluationResult:
        # Simplified check: count conditions mentioned
        problem = example.inputs.get("problem_text", "")
        explanation = run.outputs.get("explanation", "")

        # Basic heuristic: check if key terms from problem appear in explanation
        problem_terms = set(problem.split())
        explanation_terms = set(explanation.split())

        overlap = len(problem_terms & explanation_terms) / max(len(problem_terms), 1)

        return EvaluationResult(
            key="condition_usage",
            score=min(overlap * 1.5, 1.0)  # Scale up slightly
        )

    return condition_usage


def create_latex_accuracy_evaluator():
    """Evaluator for LaTeX expression accuracy"""
    @run_evaluator
    def latex_accuracy(run, example) -> EvaluationResult:
        explanation = run.outputs.get("explanation", "")

        # Check for common LaTeX patterns
        latex_patterns = [
            r"\frac", r"\int", r"\sum", r"\lim",
            r"\sin", r"\cos", r"\tan", r"\log",
            r"\sqrt", r"^{", r"_{", "$"
        ]

        latex_count = sum(1 for p in latex_patterns if p in explanation)

        # Heuristic: expect at least a few LaTeX expressions
        score = min(latex_count / 5, 1.0)

        return EvaluationResult(
            key="latex_accuracy",
            score=score
        )

    return latex_accuracy


# Create default evaluator instance
default_evaluator = MathExplanationEvaluator()