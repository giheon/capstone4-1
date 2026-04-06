from __future__ import annotations

from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.concepts.catalog import ConceptCatalog
from app.main import create_app
from tests.helpers import FakeLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = ConceptCatalog.from_project_root(PROJECT_ROOT)


def build_test_client(
    fake_client: FakeLLMClient | None = None,
    max_image_size_mb: int = 10,
) -> TestClient:
    app = create_app(
        settings=Settings(
            app_env="test",
            log_level="INFO",
            openai_api_key="test-key",
            max_image_size_mb=max_image_size_mb,
        ),
        llm_client=fake_client or FakeLLMClient(),
        catalog=CATALOG,
    )
    return TestClient(app)


class ApiTests(unittest.TestCase):
    def test_health(self) -> None:
        with build_test_client() as client:
            response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_solve_with_text_only(self) -> None:
        with build_test_client() as client:
            response = client.post("/api/v1/solve", data={"question_text": "문제"})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["difficulty"], "중")
        self.assertEqual(body["selected_model"], "gpt-5.4-mini")
        self.assertEqual(body["similar_problems"]["status"], "pending_corpus")
        self.assertTrue(body["request_id"])

    def test_solve_with_image_only(self) -> None:
        with build_test_client() as client:
            response = client.post(
                "/api/v1/solve",
                files={"image": ("problem.png", b"fake-image", "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["difficulty"], "중")

    def test_solve_with_text_and_image(self) -> None:
        with build_test_client() as client:
            response = client.post(
                "/api/v1/solve",
                data={"question_text": "문제"},
                files={"image": ("problem.png", b"fake-image", "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "1")

    def test_requires_text_or_image(self) -> None:
        with build_test_client() as client:
            response = client.post("/api/v1/solve")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error_code"], "INVALID_INPUT")

    def test_rejects_unsupported_image_type(self) -> None:
        with build_test_client() as client:
            response = client.post(
                "/api/v1/solve",
                files={"image": ("problem.gif", b"gif-data", "image/gif")},
            )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["error_code"], "UNSUPPORTED_IMAGE_TYPE")

    def test_rejects_large_image(self) -> None:
        with build_test_client(max_image_size_mb=1) as client:
            response = client.post(
                "/api/v1/solve",
                files={"image": ("problem.png", b"a" * (1024 * 1024 + 1), "image/png")},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error_code"], "IMAGE_TOO_LARGE")
