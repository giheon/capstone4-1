"""
LangGraph Workflow for Math Explanation Generation

Architecture (5 Stages):
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────┐
│ OCR Extract  │→ │ Difficulty Route │→ │ Model Selection  │→ │ Explanation Gen (3x)│→ │  Hard Gate   │
└──────────────┘  └──────────────────┘  └──────────────────┘  └─────────────────────┘  └──────────────┘
        │                 │                    │                        │                       │
        ▼                 ▼                    ▼                        ▼                       ▼
   - OCR 텍스트 추출   - routing_difficulty   - 고정 모델 선택       - 3회 병렬 호출         - 정답 다수결
   - 객관식/주관식       판정 (gpt-4.5)         (subject × difficulty)   - 프롬프트 선택       - JSON 검증
   - 과목/난이도/단원   - JSON 출력             - Gemini/OpenAI 분기   - 해설 생성            - 답 형식 검증
   - curriculum_area   - confidence / evidence                                                 - LaTeX 검증
   - major_topics                                                                            - 최종 선택
"""
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END

from .state import MathExplanationState, create_initial_state
from .nodes import (
    ocr_extraction_node,
    difficulty_routing_node,
    explanation_generation_node,
    hard_gate_node
)


def create_explanation_graph() -> StateGraph:
    """
    Create the LangGraph workflow for math explanation generation.

    Flow:
    1. OCR extraction
    2. Difficulty routing
    3. Model selection
    4. Explanation Generation (3회 병렬 호출)
    5. Hard Gate (검증 & 선택)
    """
    # Initialize graph with state schema
    workflow = StateGraph(MathExplanationState)

    # Add nodes
    workflow.add_node("ocr_extraction", ocr_extraction_node)
    workflow.add_node("difficulty_routing", difficulty_routing_node)
    workflow.add_node("generate_explanation", explanation_generation_node)
    workflow.add_node("hard_gate", hard_gate_node)

    # Set entry point
    workflow.set_entry_point("ocr_extraction")

    # Add edges (선형 흐름)
    workflow.add_edge("ocr_extraction", "difficulty_routing")
    workflow.add_edge("difficulty_routing", "generate_explanation")
    workflow.add_edge("generate_explanation", "hard_gate")
    workflow.add_edge("hard_gate", END)

    return workflow.compile()


# Create compiled graph instance
explanation_graph = create_explanation_graph()


async def run_explanation_workflow(
    image_base64: str,
    explanation_level: str = "중급",
    trace_metadata: Optional[Dict[str, Any]] = None
) -> MathExplanationState:
    """
    Run the complete explanation workflow.

    Args:
        image_base64: Base64 encoded image of math problem
        explanation_level: 초급/중급/고급 (default: 중급)

    Returns:
        Final state with generated explanation
    """
    initial_state = create_initial_state(
        image_base64=image_base64,
        explanation_level=explanation_level,
        trace_metadata=trace_metadata
    )

    # Run the graph
    final_state = await explanation_graph.ainvoke(
        initial_state,
        config={
            "run_name": "math_explanation_workflow",
            "tags": ["android-app", "math-explanation"],
            "metadata": trace_metadata or {}
        }
    )

    return final_state


async def stream_explanation_workflow(
    image_base64: str,
    explanation_level: str = "중급",
    trace_metadata: Optional[Dict[str, Any]] = None
):
    """
    Stream the explanation workflow with intermediate results.

    Yields:
        Dict with node updates as they are generated
    """
    initial_state = create_initial_state(
        image_base64=image_base64,
        explanation_level=explanation_level,
        trace_metadata=trace_metadata
    )

    # Stream through graph nodes
    async for event in explanation_graph.astream(
        initial_state,
        config={
            "run_name": "math_explanation_workflow_stream",
            "tags": ["android-app", "math-explanation", "stream"],
            "metadata": trace_metadata or {}
        }
    ):
        for node_name, node_output in event.items():

            if node_name == "ocr_extraction":
                yield {
                    "node": "ocr_extraction",
                    "status": "완료",
                    "data": {
                        "problem_text": node_output.get("problem_text", "")[:200] + "..." if len(node_output.get("problem_text", "")) > 200 else node_output.get("problem_text", ""),
                        "subject": node_output.get("subject", ""),
                        "difficulty": node_output.get("difficulty", ""),
                        "unit": node_output.get("unit", ""),
                        "question_type": node_output.get("question_type", ""),
                        "curriculum_area": node_output.get("curriculum_area", ""),
                        "major_topics": node_output.get("major_topics", [])
                    }
                }

            elif node_name == "difficulty_routing":
                yield {
                    "node": "difficulty_routing",
                    "status": "완료",
                    "data": {
                        "routing_difficulty": node_output.get("routing_difficulty", ""),
                        "routing_confidence": node_output.get("routing_confidence", 0.0),
                        "difficulty_evidence": node_output.get("difficulty_evidence", []),
                        "borderline_with": node_output.get("borderline_with", "none"),
                        "borderline_reason": node_output.get("borderline_reason", ""),
                        "selected_model": node_output.get("selected_model", "")
                    }
                }

            elif node_name == "generate_explanation":
                candidates = node_output.get("explanation_candidates", [])
                yield {
                    "node": "generate_explanation",
                    "status": "완료",
                    "data": {
                        "candidates_count": len(candidates),
                        "answers": [c.get("extracted_answer", "") for c in candidates]
                    }
                }

            elif node_name == "hard_gate":
                yield {
                    "node": "hard_gate",
                    "status": "완료",
                    "data": {
                        "majority_answer": node_output.get("majority_answer", ""),
                        "problem_review": node_output.get("problem_review", ""),
                        "condition_interpretation": node_output.get("condition_interpretation", ""),
                        "solution": node_output.get("solution", ""),
                        "answer": node_output.get("answer", ""),
                        "validation_results": node_output.get("validation_results", [])
                    }
                }

            elif node_name == "__end__":
                yield {
                    "node": "complete",
                    "status": "완료",
                    "is_complete": True
                }
