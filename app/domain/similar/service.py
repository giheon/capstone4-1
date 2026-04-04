from __future__ import annotations

from app.schemas.response import SimilarProblemsResponse


class PlaceholderSimilarProblemsService:
    async def lookup(
        self,
        concept_ids: list[str],
        flow: list[dict[str, object]],
    ) -> SimilarProblemsResponse:
        _ = concept_ids, flow
        return SimilarProblemsResponse(status="pending_corpus", items=[])
