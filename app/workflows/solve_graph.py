from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AppError
from app.domain.concepts.catalog import ConceptCatalog
from app.domain.similar.service import PlaceholderSimilarProblemsService
from app.schemas.request import ALLOWED_IMAGE_MIME_TYPES, SolveRequestInput
from app.schemas.response import SolveResponse


MODEL_BY_DIFFICULTY = {
    "상": "gpt-5.4",
    "중": "gpt-5.4-mini",
    "하": "gpt-5.4-nano",
}


class SolveState(TypedDict, total=False):
    request_id: str
    question_text: str | None
    image_bytes: bytes | None
    image_mime_type: str | None
    difficulty: str
    selected_model: str
    solution_raw: str
    answer_raw: str
    concept_ids_raw: list[str]
    flow_raw: list[dict[str, Any]]
    validated_concepts: list[dict[str, Any]]
    validated_flow: list[dict[str, Any]]
    similar_problems: dict[str, Any]
    response: dict[str, Any]
    errors: list[str]


def build_solve_graph(
    settings: Settings,
    catalog: ConceptCatalog,
    llm_client: Any,
    similar_service: PlaceholderSimilarProblemsService,
    logger: logging.Logger,
):
    async def normalize_input(state: SolveState) -> SolveState:
        try:
            normalized = SolveRequestInput(
                question_text=state.get("question_text"),
                image_bytes=state.get("image_bytes"),
                image_mime_type=state.get("image_mime_type"),
            )
        except ValidationError as exc:
            raise AppError(
                error_code="INVALID_INPUT",
                message="question_text 또는 image 중 하나는 필요합니다.",
                status_code=400,
            ) from exc

        image_bytes = normalized.image_bytes
        image_mime_type = normalized.image_mime_type

        if image_bytes:
            if image_mime_type not in ALLOWED_IMAGE_MIME_TYPES:
                raise AppError(
                    error_code="UNSUPPORTED_IMAGE_TYPE",
                    message="지원하지 않는 이미지 형식입니다.",
                    status_code=415,
                )
            if len(image_bytes) > settings.max_image_size_bytes:
                raise AppError(
                    error_code="IMAGE_TOO_LARGE",
                    message=f"이미지 크기는 {settings.max_image_size_mb}MB 이하여야 합니다.",
                    status_code=413,
                )

        return {
            "question_text": normalized.question_text,
            "image_bytes": image_bytes,
            "image_mime_type": image_mime_type,
        }

    async def classify_difficulty(state: SolveState) -> SolveState:
        request_id = state["request_id"]
        last_output = ""
        for _ in range(2):
            output = await llm_client.classify_difficulty(
                question_text=state.get("question_text"),
                image_bytes=state.get("image_bytes"),
                image_mime_type=state.get("image_mime_type"),
            )
            difficulty = output.strip()
            if difficulty in MODEL_BY_DIFFICULTY:
                return {"difficulty": difficulty}
            last_output = difficulty
        logger.warning(
            "Classifier returned invalid difficulty output.",
            extra={"request_id": request_id},
        )
        raise AppError(
            error_code="MODEL_OUTPUT_INVALID",
            message=f"난이도 분류 출력이 유효하지 않습니다: {last_output or 'empty'}",
            status_code=502,
        )

    def route_model(state: SolveState) -> SolveState:
        difficulty = state["difficulty"]
        return {"selected_model": MODEL_BY_DIFFICULTY[difficulty]}

    async def generate_solution(state: SolveState) -> SolveState:
        output = await llm_client.generate_solution(
            model=state["selected_model"],
            question_text=state.get("question_text"),
            image_bytes=state.get("image_bytes"),
            image_mime_type=state.get("image_mime_type"),
            concept_reference=catalog.prompt_reference,
        )
        return {
            "solution_raw": output.solution.strip(),
            "answer_raw": output.answer.strip(),
            "concept_ids_raw": output.concept_ids,
            "flow_raw": [step.model_dump() for step in output.flow],
        }

    def validate_concepts(state: SolveState) -> SolveState:
        request_id = state["request_id"]
        raw_concept_ids = state.get("concept_ids_raw", [])
        raw_flow = state.get("flow_raw", [])
        valid_concept_ids = catalog.merge_valid_concept_ids(raw_concept_ids, raw_flow)
        validated_concepts = catalog.build_public_concepts(valid_concept_ids)
        validated_flow = catalog.build_public_flow(raw_flow, valid_concept_ids)

        discarded_ids = set(raw_concept_ids) - set(valid_concept_ids)
        if discarded_ids:
            logger.warning(
                "Discarded invalid concept ids from model output.",
                extra={"request_id": request_id},
            )

        return {
            "validated_concepts": [concept.model_dump() for concept in validated_concepts],
            "validated_flow": [flow.model_dump() for flow in validated_flow],
        }

    async def lookup_similar_problems(state: SolveState) -> SolveState:
        validated_flow = state.get("validated_flow", [])
        concept_ids = [
            concept["concept_id"] for concept in state.get("validated_concepts", [])
        ]
        similar_problems = await similar_service.lookup(
            concept_ids=concept_ids,
            flow=validated_flow,
        )
        return {"similar_problems": similar_problems.model_dump()}

    def assemble_response(state: SolveState) -> SolveState:
        response = SolveResponse(
            request_id=state["request_id"],
            difficulty=state["difficulty"],
            selected_model=state["selected_model"],
            solution=state["solution_raw"],
            answer=state["answer_raw"],
            concepts=state.get("validated_concepts", []),
            flow=state.get("validated_flow", []),
            similar_problems=state["similar_problems"],
        )
        return {"response": response.model_dump()}

    graph = StateGraph(SolveState)
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("classify_difficulty", classify_difficulty)
    graph.add_node("route_model", route_model)
    graph.add_node("generate_solution", generate_solution)
    graph.add_node("validate_concepts", validate_concepts)
    graph.add_node("lookup_similar_problems", lookup_similar_problems)
    graph.add_node("assemble_response", assemble_response)

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "classify_difficulty")
    graph.add_edge("classify_difficulty", "route_model")
    graph.add_edge("route_model", "generate_solution")
    graph.add_edge("generate_solution", "validate_concepts")
    graph.add_edge("validate_concepts", "lookup_similar_problems")
    graph.add_edge("lookup_similar_problems", "assemble_response")
    graph.add_edge("assemble_response", END)

    return graph.compile()
