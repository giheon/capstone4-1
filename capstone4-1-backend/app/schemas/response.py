from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Difficulty = Literal["상", "중", "하"]


class ConceptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    label_ko: str
    subject: str
    chapter_id: str
    type: Literal["concept"]


class FlowStepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    concept_id: str
    label_ko: str


class SimilarProblemItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_id: str
    source_org: str
    year: int
    month: int | None = None
    subject: str
    question_no: int | None = None
    stem: str
    answer: str


class SimilarProblemsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    items: list[SimilarProblemItem] = Field(default_factory=list)


class SolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    difficulty: Difficulty
    selected_model: str
    extracted_problem: str = ""  # OCR로 추출된 문제 (LaTeX)
    solution: str
    answer: str
    concepts: list[ConceptResponse]
    flow: list[FlowStepResponse]
    similar_problems: SimilarProblemsResponse


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    error_code: str
    message: str


class FlowStepStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int
    concept_id: str


class SolverStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted_problem: str = ""  # OCR로 추출된 문제 (LaTeX)
    solution: str
    answer: str
    concept_ids: list[str] = Field(default_factory=list)
    flow: list[FlowStepStructuredOutput] = Field(default_factory=list)
