# 수능수학 AI - Math Problem Solving Platform

LangGraph 기반 수능 수학 문제 AI 해설 서비스

## 시스템 아키텍처 (3노드)

```
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
```

## 프로젝트 구조

```
capstone/
├── app/                          # Android App (Jetpack Compose)
│   └── src/main/
│       ├── java/com/example/math/
│       │   ├── MainActivity.kt
│       │   ├── navigation/
│       │   └── ui/
│       │       ├── components/
│       │       │   └── LatexView.kt      # KaTeX 렌더링
│       │       └── screens/
│       │           ├── home/
│       │           ├── camera/
│       │           └── explanation/      # AI 해설 화면
│       └── assets/
│           └── katex.html                # LaTeX 렌더링 템플릿
│
└── backend/                      # Python Backend (LangGraph + FastAPI)
    ├── main.py                   # FastAPI 서버
    ├── config.py                 # 중앙 집중식 설정 (모델, 프롬프트, 예제)
    ├── requirements.txt
    └── graph/
        ├── __init__.py
        ├── state.py              # LangGraph State 스키마
        ├── nodes.py              # 3개 노드 (OCR, Gen, HardGate)
        └── workflow.py           # 워크플로우 정의
```

## 기능

### Android App
- 수학 문제 촬영 (CameraX)
- 해설 수준 선택 (초급/중급/고급)
- 실시간 스트리밍 해설 표시
- LaTeX 수식 렌더링 (KaTeX)
- 3단계 해설 구조: 문제 리뷰 → 조건 해석 → 문제 풀이

### Backend (LangGraph)

#### Node 1: OCR + 라우팅
- GPT-4o Vision으로 문제 텍스트 추출
- 객관식/주관식 판별
- 과목 분류: 미적분, 확률과통계, 기하
- 난이도 분류: 쉬움, 보통, 어려움, 킬러
- 단원 분류 (과목별 10개 단원)
- 모델 라우팅 (과목×난이도 → 모델)

#### Node 2: 해설 생성 (3회 병렬)
- 9개 프롬프트 (과목 3 × 해설수준 3)
- Few-shot 예제 주입 (단원별 3개)
- 동일 프롬프트로 3회 병렬 호출
- LaTeX 수식 사용

#### Node 3: Hard Gate (코드 레벨)
- 정답 다수결 (2개 이상 일치)
- JSON 구조 검증
- 답 형식 검증 (question_type 일치)
- LaTeX 문법 검증
- 최종 후보 선택 (랜덤)

## 설정 (config.py)

모든 설정을 한 곳에서 관리:

```python
# 모델 라우팅 (과목 × 난이도 → 모델)
MODEL_ROUTING = {
    ("미적분", "킬러"): "gpt-4o",
    ("미적분", "보통"): "gpt-4o-mini",
    # ...
}

# 해설 프롬프트 (과목 × 해설수준)
EXPLANATION_PROMPTS = {
    ("미적분", "초급"): "...",
    ("미적분", "중급"): "...",
    # ...
}

# Few-shot 예제 (과목 × 단원)
FEW_SHOT_EXAMPLES = {
    "미적분": {
        "미분법": [{...}, {...}, {...}],
        # ...
    }
}
```

## 실행 방법

### Backend
```bash
cd backend

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
# LangSmith 추적을 쓰려면 LANGSMITH_API_KEY도 입력

# 의존성 설치
pip install -r requirements.txt

# 실행
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

LangSmith 연동에는 LangSmith 계정의 API 키가 필요합니다. `backend/.env`에 아래 값을 설정하면 백엔드의 LangGraph/LCEL 실행이 `math-explanation` 프로젝트로 트래킹됩니다.

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=math-explanation
```

Android 에뮬레이터는 기본적으로 `http://10.0.2.2:8000`의 백엔드에 연결합니다. 실제 기기에서 테스트할 경우 [app/build.gradle.kts](/Users/seonmain10/Desktop/capstone/app/build.gradle.kts)의 `API_BASE_URL`을 개발 머신 IP로 변경해야 합니다.

### Android App
```bash
# Android Studio에서 프로젝트 열기
# Run 'app' 실행
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 헬스체크 |
| POST | `/explain` | 전체 해설 생성 |
| POST | `/explain/stream` | SSE 스트리밍 |
| POST | `/explain/upload` | 이미지 업로드 + 스트리밍 |
| GET | `/meta/subjects` | 과목 목록 |
| GET | `/meta/units/{subject}` | 단원 목록 |
| GET | `/meta/levels` | 해설 수준 목록 |
| GET | `/meta/config` | 현재 설정 조회 |
| GET | `/meta/routing` | 모델 라우팅 조회 |

## 요청/응답 형식

### 요청
```json
{
    "image_base64": "...",
    "explanation_level": "중급"
}
```

### 응답
```json
{
    "problem_text": "함수 f(x) = ...",
    "question_type": "subjective",
    "subject": "미적분",
    "difficulty": "보통",
    "unit": "미분법",
    "selected_model": "gpt-4o-mini",
    "problem_review": "[1. 문제 리뷰] 내용",
    "condition_interpretation": "[2. 조건 해석] 내용",
    "solution": "[3. 문제 풀이] 내용... 답: 2",
    "answer": "2",
    "majority_answer": "2",
    "is_complete": true
}
```

## 기술 스택

### Frontend
- Kotlin
- Jetpack Compose
- Material3
- CameraX
- KaTeX (WebView)

### Backend
- Python 3.10+
- LangGraph / LangChain
- FastAPI
- OpenAI GPT-4o / GPT-4o-mini
- LCEL (LangChain Expression Language)

## 검증 기준 (Hard Gate)

| 검증 | 방법 |
|------|------|
| 정답 다수결 | 3개 후보 중 2개 이상 일치 |
| JSON 구조 | 필수 필드 존재 여부 |
| 답 형식 | question_type과 추출된 답 일치 |
| LaTeX 문법 | 괄호 매칭 검사 |

## 라이선스

This project is for educational purposes.
