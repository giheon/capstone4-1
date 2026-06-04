"""
FastAPI Server for Math Explanation Generation

Architecture (3 Nodes):
- Node 1: OCR + Routing (통합)
- Node 2: Explanation Generation (3회 병렬 호출)
- Node 3: Hard Gate (검증 & 선택)

Endpoints:
- GET  /health              : Health check
- POST /explain             : Generate explanation (non-streaming)
- POST /explain/stream      : Generate explanation with SSE streaming
- POST /explain/upload      : Upload image and stream explanation
- GET  /meta/subjects       : Get subject list
- GET  /meta/units/{subject}: Get units for subject
- GET  /meta/levels         : Get explanation levels
- GET  /meta/config         : Get current configuration
"""
import json
import base64
from typing import Optional, Literal, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph.workflow import (
    run_explanation_workflow,
    stream_explanation_workflow
)

from config import (
    UNITS,
    MODEL_ROUTING,
    EXPLANATION_PROMPTS,
    PARALLEL_CALL_COUNT
)

app = FastAPI(
    title="수능수학 AI 해설 API",
    description="LangGraph 기반 수학 문제 해설 생성 서비스 (3노드 아키텍처)",
    version="2.0.0"
)

# CORS settings for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════════════

class ExplanationRequest(BaseModel):
    """Request model for explanation generation"""
    image_base64: str
    explanation_level: Literal["초급", "중급", "고급"] = "중급"
    client_session_id: Optional[str] = None
    client_source: str = "android"
    client_metadata: Dict[str, Any] = Field(default_factory=dict)


class ExplanationResponse(BaseModel):
    """Response model for complete explanation"""
    # OCR + 라우팅 결과
    problem_text: str
    question_type: str  # objective/subjective
    subject: str  # 확률과통계/미적분/기하
    difficulty: str  # 쉬움/보통/어려움/킬러
    unit: str  # 단원
    selected_model: str  # 사용된 모델

    # 최종 해설
    problem_review: str  # [1. 문제 리뷰]
    condition_interpretation: str  # [2. 조건 해석]
    solution: str  # [3. 문제 풀이]
    answer: str  # 최종 답

    # 메타 정보
    majority_answer: str  # 다수결 답
    is_complete: bool


def build_trace_metadata(request: ExplanationRequest) -> Dict[str, Any]:
    """Build LangSmith-safe metadata for an Android test request."""
    return {
        "client_source": request.client_source,
        "client_session_id": request.client_session_id,
        "explanation_level": request.explanation_level,
        **request.client_metadata
    }


class StreamChunk(BaseModel):
    """Model for streaming chunks"""
    node: str
    status: str
    data: dict
    is_complete: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "math-explanation-api",
        "version": "2.0.0",
        "architecture": "3-node (OCR+Routing → Generation(3x) → HardGate)",
        "parallel_calls": PARALLEL_CALL_COUNT
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/explain", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest):
    """
    Generate a complete math problem explanation.

    Args:
        request: ExplanationRequest with base64 encoded image and explanation_level

    Returns:
        Complete explanation with all sections
    """
    try:
        # Run the LangGraph workflow
        result = await run_explanation_workflow(
            image_base64=request.image_base64,
            explanation_level=request.explanation_level,
            trace_metadata=build_trace_metadata(request)
        )

        # Check for errors
        if result.get("error_message"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error_message")
            )

        return ExplanationResponse(
            problem_text=result.get("problem_text", ""),
            question_type=result.get("question_type", "subjective"),
            subject=result.get("subject", "미적분"),
            difficulty=result.get("difficulty", "보통"),
            unit=result.get("unit", ""),
            selected_model=result.get("selected_model", ""),
            problem_review=result.get("problem_review", ""),
            condition_interpretation=result.get("condition_interpretation", ""),
            solution=result.get("solution", ""),
            answer=result.get("answer", ""),
            majority_answer=result.get("majority_answer", ""),
            is_complete=result.get("is_complete", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/stream")
async def stream_explanation(request: ExplanationRequest):
    """
    Stream explanation generation using Server-Sent Events.

    Args:
        request: ExplanationRequest with base64 encoded image and explanation_level

    Returns:
        SSE stream with node updates
    """
    async def event_generator():
        try:
            async for chunk in stream_explanation_workflow(
                image_base64=request.image_base64,
                explanation_level=request.explanation_level,
                trace_metadata=build_trace_metadata(request)
            ):
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
async def upload_and_explain(
    file: UploadFile = File(...),
    explanation_level: Literal["초급", "중급", "고급"] = Form(default="중급")
):
    """
    Upload an image file and generate explanation.

    Args:
        file: Uploaded image file
        explanation_level: 해설 수준 (초급/중급/고급)

    Returns:
        SSE stream with node updates
    """
    try:
        # Read and encode image
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

        async def event_generator():
            async for chunk in stream_explanation_workflow(
                image_base64=image_base64,
                explanation_level=explanation_level,
                trace_metadata={
                    "client_source": "upload",
                    "client_session_id": None,
                    "explanation_level": explanation_level,
                    "filename": file.filename
                }
            ):
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


# ═══════════════════════════════════════════════════════════════════════════
# Meta Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/meta/subjects")
async def get_subjects():
    """과목 목록 조회"""
    return {
        "subjects": list(UNITS.keys())
    }


@app.get("/meta/units/{subject}")
async def get_units(subject: str):
    """과목별 단원 목록 조회"""
    if subject not in UNITS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown subject: {subject}. Available: {list(UNITS.keys())}"
        )

    return {
        "subject": subject,
        "units": UNITS[subject]
    }


@app.get("/meta/levels")
async def get_explanation_levels():
    """해설 수준 목록 조회"""
    return {
        "levels": [
            {"value": "초급", "description": "기본 개념 중심의 상세한 설명"},
            {"value": "중급", "description": "핵심 풀이 과정 중심"},
            {"value": "고급", "description": "간결한 풀이와 심화 내용"}
        ]
    }


@app.get("/meta/config")
async def get_config():
    """현재 설정 조회"""
    return {
        "subjects": list(UNITS.keys()),
        "difficulties": ["쉬움", "보통", "어려움", "킬러"],
        "explanation_levels": ["초급", "중급", "고급"],
        "parallel_call_count": PARALLEL_CALL_COUNT,
        "model_routing": {
            f"{subject}/{difficulty}": model
            for (subject, difficulty), model in MODEL_ROUTING.items()
        },
        "prompt_count": len(EXPLANATION_PROMPTS),
        "units_per_subject": {
            subject: units for subject, units in UNITS.items()
        }
    }


@app.get("/meta/routing")
async def get_model_routing():
    """모델 라우팅 설정 조회"""
    return {
        "routing": [
            {
                "subject": subject,
                "difficulty": difficulty,
                "model": model
            }
            for (subject, difficulty), model in MODEL_ROUTING.items()
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test Endpoint (개발용)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/test/mock")
async def test_mock_explanation(
    explanation_level: Literal["초급", "중급", "고급"] = "중급"
):
    """
    테스트용 목업 해설 생성 (실제 API 호출 없음)
    """
    return {
        "problem_text": "함수 $f(x) = x^3 - 3x^2 + 2$의 극댓값을 구하시오.",
        "question_type": "subjective",
        "subject": "미적분",
        "difficulty": "보통",
        "unit": "미분법",
        "selected_model": "gpt-4o-mini",
        "problem_review": "3차 함수의 극값을 구하는 문제입니다. $f'(x) = 0$인 점에서 극값 후보를 찾고, 부호 변화를 확인합니다.",
        "condition_interpretation": "$f(x) = x^3 - 3x^2 + 2$는 3차 함수이며, 미분하면 $f'(x) = 3x^2 - 6x$입니다.",
        "solution": "$f'(x) = 3x^2 - 6x = 3x(x-2) = 0$에서 $x = 0$ 또는 $x = 2$입니다.\n\n$x = 0$에서 $f'(x)$의 부호가 양에서 음으로 바뀌므로 극대입니다.\n\n$f(0) = 0 - 0 + 2 = 2$\n\n따라서 극댓값은 2입니다.\n\n답: 2",
        "answer": "2",
        "majority_answer": "2",
        "is_complete": True,
        "explanation_level": explanation_level
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
