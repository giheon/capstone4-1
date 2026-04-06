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


# 난이도별 모델 매핑
DIFFICULTY_MODEL_MAP = {
    "상": "gpt-4o",      # 어려운 문제: 최고 성능 모델
    "중": "gpt-4o",      # 중간 문제: 고성능 모델
    "하": "gpt-4o-mini", # 쉬운 문제: 경량 모델 (빠르고 저렴)
}
DEFAULT_MODEL = "gpt-4o"  # 분류 실패 시 기본값


class SolveState(TypedDict, total=False):
    request_id: str
    question_text: str | None
    image_bytes: bytes | None
    image_mime_type: str | None
    difficulty: str
    selected_model: str
    extracted_problem_raw: str  # OCR 추출 문제
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
        """GPT-4o를 사용하여 문제 난이도를 상/중/하로 분류"""
        try:
            difficulty = await llm_client.classify_difficulty(
                question_text=state.get("question_text"),
                image_bytes=state.get("image_bytes"),
                image_mime_type=state.get("image_mime_type"),
            )

            # 유효한 난이도인지 확인
            if difficulty not in ["상", "중", "하"]:
                logger.warning(
                    f"Invalid difficulty '{difficulty}' returned, using default '중'",
                    extra={"request_id": state["request_id"]},
                )
                difficulty = "중"

            logger.info(
                f"문제 난이도 분류 결과: {difficulty}",
                extra={"request_id": state["request_id"]},
            )

        except Exception as exc:
            logger.warning(
                f"Difficulty classification failed: {exc}, using default '중'",
                extra={"request_id": state["request_id"]},
            )
            difficulty = "중"

        return {"difficulty": difficulty}

    def select_model(state: SolveState) -> SolveState:
        """난이도에 따라 적절한 모델 선택"""
        difficulty = state.get("difficulty", "중")
        selected_model = DIFFICULTY_MODEL_MAP.get(difficulty, DEFAULT_MODEL)

        logger.info(
            f"난이도 '{difficulty}' → 모델 '{selected_model}' 선택",
            extra={"request_id": state["request_id"]},
        )

        return {"selected_model": selected_model}

    async def generate_solution(state: SolveState) -> SolveState:
        output = await llm_client.generate_solution(
            model=state["selected_model"],
            question_text=state.get("question_text"),
            image_bytes=state.get("image_bytes"),
            image_mime_type=state.get("image_mime_type"),
            concept_reference=catalog.prompt_reference,
        )
        return {
            "extracted_problem_raw": output.extracted_problem.strip() if output.extracted_problem else "",
            "solution_raw": output.solution.strip(),
            "answer_raw": output.answer.strip(),
            "concept_ids_raw": output.concept_ids or [],
            "flow_raw": [step.model_dump() for step in output.flow] if output.flow else [],
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
            extracted_problem=state.get("extracted_problem_raw", ""),
            solution=state["solution_raw"],
            answer=state["answer_raw"],
            concepts=state.get("validated_concepts", []),
            flow=state.get("validated_flow", []),
            similar_problems=state["similar_problems"],
        )
        return {"response": response.model_dump()}

    graph = StateGraph(SolveState)
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("classify_difficulty", classify_difficulty)  # 난이도 분류
    graph.add_node("select_model", select_model)                # 모델 선택
    graph.add_node("generate_solution", generate_solution)
    graph.add_node("validate_concepts", validate_concepts)
    graph.add_node("lookup_similar_problems", lookup_similar_problems)
    graph.add_node("assemble_response", assemble_response)

    # 워크플로우: 입력정규화 → 난이도분류 → 모델선택 → 풀이생성 → 개념검증 → 유사문제 → 응답조립
    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "classify_difficulty")
    graph.add_edge("classify_difficulty", "select_model")
    graph.add_edge("select_model", "generate_solution")
    graph.add_edge("generate_solution", "validate_concepts")
    graph.add_edge("validate_concepts", "lookup_similar_problems")
    graph.add_edge("lookup_similar_problems", "assemble_response")
    graph.add_edge("assemble_response", END)

    return graph.compile()
