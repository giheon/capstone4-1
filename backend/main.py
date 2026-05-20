"""
FastAPI Server for Math Explanation Generation

Endpoints:
- POST /explain: Generate explanation (non-streaming)
- POST /explain/stream: Generate explanation with SSE streaming
- GET /health: Health check
"""
import json
import base64
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph.workflow import (
    run_explanation_workflow,
    stream_explanation_workflow
)

app = FastAPI(
    title="수능수학 AI 해설 API",
    description="LangGraph 기반 수학 문제 해설 생성 서비스",
    version="1.0.0"
)

# CORS settings for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExplanationRequest(BaseModel):
    """Request model for explanation generation"""
    image_base64: str


class ExplanationResponse(BaseModel):
    """Response model for complete explanation"""
    problem_text: str
    problem_type: str
    difficulty: str
    section_review: str
    section_interpret: str
    section_solve: str
    answer: str
    quality_score: float


class StreamChunk(BaseModel):
    """Model for streaming chunks"""
    section: str
    title: str
    content: str
    is_complete: bool


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "math-explanation-api"}


@app.post("/explain", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest):
    """
    Generate a complete math problem explanation.

    Args:
        request: ExplanationRequest with base64 encoded image

    Returns:
        Complete explanation with all sections
    """
    try:
        # Run the LangGraph workflow
        result = await run_explanation_workflow(request.image_base64)

        return ExplanationResponse(
            problem_text=result.get("problem_text", ""),
            problem_type=result.get("problem_type", ""),
            difficulty=result.get("difficulty", "일반"),
            section_review=result.get("section_review", ""),
            section_interpret=result.get("section_interpret", ""),
            section_solve=result.get("section_solve", ""),
            answer=result.get("answer", ""),
            quality_score=result.get("quality_score", 0.0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/stream")
async def stream_explanation(request: ExplanationRequest):
    """
    Stream explanation generation using Server-Sent Events.

    Args:
        request: ExplanationRequest with base64 encoded image

    Returns:
        SSE stream with section updates
    """
    async def event_generator():
        try:
            async for chunk in stream_explanation_workflow(request.image_base64):
                yield {
                    "event": "message",
                    "data": json.dumps(chunk, ensure_ascii=False)
                }

            # Send completion event
            yield {
                "event": "complete",
                "data": json.dumps({"status": "done"})
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


@app.post("/explain/upload")
async def upload_and_explain(file: UploadFile = File(...)):
    """
    Upload an image file and generate explanation.

    Args:
        file: Uploaded image file

    Returns:
        SSE stream with section updates
    """
    try:
        # Read and encode image
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

        async def event_generator():
            async for chunk in stream_explanation_workflow(image_base64):
                yield {
                    "event": "message",
                    "data": json.dumps(chunk, ensure_ascii=False)
                }

            yield {
                "event": "complete",
                "data": json.dumps({"status": "done"})
            }

        return EventSourceResponse(event_generator())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)