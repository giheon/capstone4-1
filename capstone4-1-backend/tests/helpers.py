from __future__ import annotations

from app.schemas.response import FlowStepStructuredOutput, SolverStructuredOutput


class FakeLLMClient:
    def __init__(
        self,
        difficulties: list[str] | None = None,
        solver_output: SolverStructuredOutput | None = None,
    ) -> None:
        self._difficulties = difficulties or ["중"]
        self._solver_output = solver_output or SolverStructuredOutput(
            solution="핵심 식을 정리한 뒤 바로 계산한다.",
            answer="1",
            concept_ids=["MATH1_CH01_CONCEPT_ROOT_NTH"],
            flow=[
                FlowStepStructuredOutput(
                    order=1,
                    concept_id="MATH1_CH01_CONCEPT_ROOT_NTH",
                )
            ],
        )
        self.classify_calls = 0
        self.generate_calls: list[str] = []

    async def classify_difficulty(
        self,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
    ) -> str:
        _ = question_text, image_bytes, image_mime_type
        index = min(self.classify_calls, len(self._difficulties) - 1)
        self.classify_calls += 1
        return self._difficulties[index]

    async def generate_solution(
        self,
        model: str,
        question_text: str | None,
        image_bytes: bytes | None,
        image_mime_type: str | None,
        concept_reference: str,
    ) -> SolverStructuredOutput:
        _ = question_text, image_bytes, image_mime_type, concept_reference
        self.generate_calls.append(model)
        return self._solver_output
