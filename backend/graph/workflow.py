"""
LangGraph Workflow for Math Explanation Generation

Architecture (3 Nodes):
┌─────────────────┐   ┌─────────────────────┐   ┌─────────────────┐
│  OCR + Routing  │ → │ Explanation Gen (3x)│ → │    Hard Gate    │
│    (Node 1)     │   │      (Node 2)       │   │    (Node 3)     │
└─────────────────┘   └─────────────────────┘   └─────────────────┘
        │                      │                        │
        ▼                      ▼                        ▼
   - OCR 텍스트 추출      - 3회 병렬 호출          - 정답 다수결
   - 객관식/주관식       - Few-shot 적용          - JSON 검증
   - 과목/난이도/단원    - 해설 생성              - 답 형식 검증
   - 모델 라우팅                                  - LaTeX 검증
                                                  - 최종 선택
"""
from langgraph.graph import StateGraph, END

from .state import MathExplanationState, create_initial_state
from .nodes import (
    ocr_routing_node,
    explanation_generation_node,
    hard_gate_node
)


def create_explanation_graph() -> StateGraph:
    """
    Create the LangGraph workflow for math explanation generation.

    Flow:
    1. OCR + Routing (통합)
    2. Explanation Generation (3회 병렬 호출)
    3. Hard Gate (검증 & 선택)
    """
    # Initialize graph with state schema
    workflow = StateGraph(MathExplanationState)

    # Add nodes (3개)
    workflow.add_node("ocr_routing", ocr_routing_node)
    workflow.add_node("generate_explanation", explanation_generation_node)
    workflow.add_node("hard_gate", hard_gate_node)

    # Set entry point
    workflow.set_entry_point("ocr_routing")

    # Add edges (선형 흐름)
    workflow.add_edge("ocr_routing", "generate_explanation")
    workflow.add_edge("generate_explanation", "hard_gate")
    workflow.add_edge("hard_gate", END)

    return workflow.compile()


# Create compiled graph instance
explanation_graph = create_explanation_graph()


async def run_explanation_workflow(
    image_base64: str,
    explanation_level: str = "중급"
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
        explanation_level=explanation_level
    )

    # Run the graph
    final_state = await explanation_graph.ainvoke(initial_state)

    return final_state


async def stream_explanation_workflow(
    image_base64: str,
    explanation_level: str = "중급"
):
    """
    Stream the explanation workflow with intermediate results.

    Yields:
        Dict with node updates as they are generated
    """
    initial_state = create_initial_state(
        image_base64=image_base64,
        explanation_level=explanation_level
    )

    # Stream through graph nodes
    async for event in explanation_graph.astream(initial_state):
        for node_name, node_output in event.items():

            if node_name == "ocr_routing":
                yield {
                    "node": "ocr_routing",
                    "status": "완료",
                    "data": {
                        "problem_text": node_output.get("problem_text", "")[:200] + "..." if len(node_output.get("problem_text", "")) > 200 else node_output.get("problem_text", ""),
                        "subject": node_output.get("subject", ""),
                        "difficulty": node_output.get("difficulty", ""),
                        "unit": node_output.get("unit", ""),
                        "question_type": node_output.get("question_type", ""),
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
