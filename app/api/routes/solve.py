from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.schemas.response import SolveResponse


router = APIRouter(prefix="/api/v1", tags=["solve"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/solve", response_model=SolveResponse)
async def solve(
    request: Request,
    question_text: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
) -> SolveResponse:
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    if image is not None:
        image_bytes = await image.read()
        image_mime_type = image.content_type
        await image.close()

    result = await request.app.state.solve_graph.ainvoke(
        {
            "request_id": request.state.request_id,
            "question_text": question_text,
            "image_bytes": image_bytes,
            "image_mime_type": image_mime_type,
            "errors": [],
        }
    )
    return SolveResponse.model_validate(result["response"])
