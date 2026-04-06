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

    response = SolveResponse.model_validate(result["response"])

    # 터미널에 결과 출력
    difficulty_labels = {"상": "어려움", "중": "보통", "하": "쉬움"}
    print("\n" + "="*60)
    print("📝 수학 문제 풀이 결과")
    print("="*60)
    print(f"\n🔍 OCR 추출 문제:\n{response.extracted_problem}")
    print(f"\n📈 난이도: {response.difficulty} ({difficulty_labels.get(response.difficulty, '?')})")
    print(f"\n📊 사용 모델: {response.selected_model}")
    print(f"\n📖 풀이 과정:\n{response.solution}")
    print(f"\n✅ 정답: {response.answer}")
    if response.concepts:
        concepts_str = ", ".join([c.label_ko for c in response.concepts])
        print(f"\n📚 사용된 개념: {concepts_str}")
    print("\n" + "="*60 + "\n")

    return response
