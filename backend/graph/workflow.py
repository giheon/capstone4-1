"""
LangGraph Workflow for Math Explanation Generation

Workflow:
┌─────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
│   OCR   │ → │  Difficulty  │ → │ Explanation│ → │   Quality    │
│         │   │Classification│   │ Generation │   │  Evaluation  │
└─────────┘   └──────────────┘   └────────────┘   └──────────────┘
                                        ↑               │
                                        │    ┌──────────┴──────────┐
                                        │    ↓                     ↓
                                  ┌───────────┐            ┌───────────┐
                                  │Regenerate │            │  Complete │
                                  └───────────┘            └───────────┘
"""
from langgraph.graph import StateGraph, END

from .state import MathExplanationState
from .nodes import (
    ocr_node,
    difficulty_classification_node,
    explanation_generation_node,
    quality_evaluation_node,
    regeneration_node,
    should_regenerate
)


def create_explanation_graph() -> StateGraph:
    """
    Create the LangGraph workflow for math explanation generation.
    """
    # Initialize graph with state schema
    workflow = StateGraph(MathExplanationState)

    # Add nodes
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("classify_difficulty", difficulty_classification_node)
    workflow.add_node("generate_explanation", explanation_generation_node)
    workflow.add_node("evaluate_quality", quality_evaluation_node)
    workflow.add_node("regenerate", regeneration_node)

    # Set entry point
    workflow.set_entry_point("ocr")

    # Add edges (linear flow)
    workflow.add_edge("ocr", "classify_difficulty")
    workflow.add_edge("classify_difficulty", "generate_explanation")
    workflow.add_edge("generate_explanation", "evaluate_quality")

    # Add conditional edge for quality gate
    workflow.add_conditional_edges(
        "evaluate_quality",
        should_regenerate,
        {
            "regenerate": "regenerate",
            "complete": END
        }
    )

    # Regeneration loops back to quality evaluation
    workflow.add_edge("regenerate", "evaluate_quality")

    return workflow.compile()


# Create compiled graph instance
explanation_graph = create_explanation_graph()


async def run_explanation_workflow(image_base64: str) -> MathExplanationState:
    """
    Run the complete explanation workflow.

    Args:
        image_base64: Base64 encoded image of math problem

    Returns:
        Final state with generated explanation
    """
    initial_state: MathExplanationState = {
        "image_base64": image_base64,
        "problem_text": "",
        "problem_type": "",
        "difficulty": "일반",
        "selected_model": "gpt-4o-mini",
        "section_review": "",
        "section_interpret": "",
        "section_solve": "",
        "answer": "",
        "quality_score": 0.0,
        "quality_feedback": "",
        "retry_count": 0,
        "max_retries": 2,
        "current_section": "review",
        "stream_content": "",
        "is_formula": False,
        "is_complete": False,
        "error_message": None
    }

    # Run the graph
    final_state = await explanation_graph.ainvoke(initial_state)

    return final_state


async def stream_explanation_workflow(image_base64: str):
    """
    Stream the explanation workflow with intermediate results.

    Yields:
        Dict with section updates as they are generated
    """
    initial_state: MathExplanationState = {
        "image_base64": image_base64,
        "problem_text": "",
        "problem_type": "",
        "difficulty": "일반",
        "selected_model": "gpt-4o-mini",
        "section_review": "",
        "section_interpret": "",
        "section_solve": "",
        "answer": "",
        "quality_score": 0.0,
        "quality_feedback": "",
        "retry_count": 0,
        "max_retries": 2,
        "current_section": "review",
        "stream_content": "",
        "is_formula": False,
        "is_complete": False,
        "error_message": None
    }

    # Stream through graph nodes
    async for event in explanation_graph.astream(initial_state):
        # Extract node name and output
        for node_name, node_output in event.items():
            if node_name == "generate_explanation":
                # Yield each section as it's generated
                if node_output.get("section_review"):
                    yield {
                        "section": "review",
                        "title": "문제 리뷰",
                        "content": node_output["section_review"],
                        "is_complete": False
                    }

                if node_output.get("section_interpret"):
                    yield {
                        "section": "interpret",
                        "title": "조건 해석",
                        "content": node_output["section_interpret"],
                        "is_complete": False
                    }

                if node_output.get("section_solve"):
                    yield {
                        "section": "solve",
                        "title": "문제 풀이",
                        "content": node_output["section_solve"],
                        "is_complete": False
                    }

                if node_output.get("answer"):
                    yield {
                        "section": "answer",
                        "title": "정답",
                        "content": node_output["answer"],
                        "is_complete": False
                    }

            elif node_name == "__end__":
                yield {
                    "section": "complete",
                    "title": "",
                    "content": "",
                    "is_complete": True
                }