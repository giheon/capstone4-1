from __future__ import annotations

from pathlib import Path
import unittest

from app.core.config import Settings
from app.domain.concepts.catalog import ConceptCatalog
from app.domain.similar.service import PlaceholderSimilarProblemsService
from app.schemas.response import FlowStepStructuredOutput, SolverStructuredOutput
from app.workflows.solve_graph import build_solve_graph
from tests.helpers import FakeLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SolveWorkflowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = ConceptCatalog.from_project_root(PROJECT_ROOT)
        cls.settings = Settings(
            app_env="test",
            log_level="INFO",
            openai_api_key="test-key",
            max_image_size_mb=10,
        )

    async def test_routes_to_model_by_difficulty(self) -> None:
        fake_client = FakeLLMClient(difficulties=["중"])
        graph = build_solve_graph(
            settings=self.settings,
            catalog=self.catalog,
            llm_client=fake_client,
            similar_service=PlaceholderSimilarProblemsService(),
            logger=__import__("logging").getLogger("test"),
        )

        result = await graph.ainvoke(
            {
                "request_id": "req-1",
                "question_text": "간단한 문제",
                "image_bytes": None,
                "image_mime_type": None,
                "errors": [],
            }
        )

        self.assertEqual(result["response"]["selected_model"], "gpt-5.4-mini")
        self.assertEqual(fake_client.generate_calls, ["gpt-5.4-mini"])

    async def test_retries_invalid_difficulty_output_once(self) -> None:
        fake_client = FakeLLMClient(difficulties=["어려움", "하"])
        graph = build_solve_graph(
            settings=self.settings,
            catalog=self.catalog,
            llm_client=fake_client,
            similar_service=PlaceholderSimilarProblemsService(),
            logger=__import__("logging").getLogger("test"),
        )

        result = await graph.ainvoke(
            {
                "request_id": "req-2",
                "question_text": "분류 재시도 문제",
                "image_bytes": None,
                "image_mime_type": None,
                "errors": [],
            }
        )

        self.assertEqual(fake_client.classify_calls, 2)
        self.assertEqual(result["response"]["difficulty"], "하")
        self.assertEqual(result["response"]["selected_model"], "gpt-5.4-nano")

    async def test_filters_invalid_concepts_and_flow(self) -> None:
        fake_client = FakeLLMClient(
            difficulties=["상"],
            solver_output=SolverStructuredOutput(
                solution="핵심 개념만 써서 푼다.",
                answer="3",
                concept_ids=[
                    "MATH1_CH01_CONCEPT_ROOT_NTH",
                    "INVALID_CONCEPT",
                ],
                flow=[
                    FlowStepStructuredOutput(order=1, concept_id="INVALID_CONCEPT"),
                    FlowStepStructuredOutput(
                        order=2,
                        concept_id="MATH1_CH01_CONCEPT_ROOT_NTH",
                    ),
                ],
            ),
        )
        graph = build_solve_graph(
            settings=self.settings,
            catalog=self.catalog,
            llm_client=fake_client,
            similar_service=PlaceholderSimilarProblemsService(),
            logger=__import__("logging").getLogger("test"),
        )

        result = await graph.ainvoke(
            {
                "request_id": "req-3",
                "question_text": "개념 검증 문제",
                "image_bytes": None,
                "image_mime_type": None,
                "errors": [],
            }
        )

        self.assertEqual(len(result["response"]["concepts"]), 1)
        self.assertEqual(
            result["response"]["concepts"][0]["concept_id"],
            "MATH1_CH01_CONCEPT_ROOT_NTH",
        )
        self.assertEqual(len(result["response"]["flow"]), 1)
        self.assertEqual(result["response"]["flow"][0]["order"], 1)
