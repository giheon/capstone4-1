"""
LangGraph components for math explanation generation.

Architecture (3 Nodes):
1. OCR + Routing (통합) - LCEL
2. Explanation Generation (3회 병렬 호출) - LCEL
3. Hard Gate (검증 & 선택) - 순수 코드
"""
from .state import (
    MathExplanationState,
    ExplanationCandidate,
    ValidationResult,
    create_initial_state
)
from .workflow import (
    explanation_graph,
    run_explanation_workflow,
    stream_explanation_workflow,
    create_explanation_graph
)
from .nodes import (
    ocr_routing_node,
    explanation_generation_node,
    hard_gate_node,
    extract_answer,
    validate_json_structure,
    validate_answer_format,
    validate_latex
)

__all__ = [
    # State
    "MathExplanationState",
    "ExplanationCandidate",
    "ValidationResult",
    "create_initial_state",

    # Workflow
    "explanation_graph",
    "run_explanation_workflow",
    "stream_explanation_workflow",
    "create_explanation_graph",

    # Nodes
    "ocr_routing_node",
    "explanation_generation_node",
    "hard_gate_node",

    # Utilities
    "extract_answer",
    "validate_json_structure",
    "validate_answer_format",
    "validate_latex"
]
