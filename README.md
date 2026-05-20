# 수능수학 AI - Math Problem Solving Platform

LangGraph 기반 수능 수학 문제 AI 해설 서비스

## 프로젝트 구조

```
capstone/
├── app/                          # Android App (Jetpack Compose)
│   └── src/main/
│       ├── java/com/example/math/
│       │   ├── MainActivity.kt
│       │   ├── navigation/
│       │   │   └── NavGraph.kt
│       │   └── ui/
│       │       ├── components/
│       │       │   └── LatexView.kt      # KaTeX 렌더링
│       │       ├── screens/
│       │       │   ├── home/
│       │       │   ├── camera/
│       │       │   └── explanation/      # AI 해설 화면
│       │       └── theme/
│       └── assets/
│           └── katex.html                # LaTeX 렌더링 템플릿
│
└── backend/                      # Python Backend (LangGraph + FastAPI)
    ├── main.py                   # FastAPI 서버
    ├── requirements.txt
    ├── graph/
    │   ├── state.py              # LangGraph State 스키마
    │   ├── nodes.py              # OCR, 해설생성, 품질평가 노드
    │   └── workflow.py           # 워크플로우 정의
    ├── prompts/
    │   └── explanation.py        # 프롬프트 템플릿
    └── evaluators/
        └── quality.py            # LangSmith 품질 평가
```

## 기능

### Android App
- 수학 문제 촬영 (CameraX)
- 실시간 스트리밍 해설 표시
- LaTeX 수식 렌더링 (KaTeX)
- 3단계 해설 구조: 문제 리뷰 → 조건 해석 → 문제 풀이

### Backend (LangGraph)
- **OCR**: GPT-4o Vision으로 문제 텍스트 추출
- **Hard Gate**: 난이도 기반 모델 라우팅 (킬러→GPT-4o, 일반→GPT-4o-mini)
- **해설 생성**: 재현 가능한 사고 과정 제공
- **Soft Quality Gate**: LangSmith 기반 품질 평가 (75점 미만 시 재생성)

## 실행 방법

### Android App
```bash
# Android Studio에서 프로젝트 열기
# Run 'app' 실행
```

### Backend
```bash
cd backend

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 실행
./run.sh
# 또는
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 헬스체크 |
| POST | `/explain` | 전체 해설 생성 |
| POST | `/explain/stream` | SSE 스트리밍 |
| POST | `/explain/upload` | 이미지 업로드 + 스트리밍 |

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
- LangSmith (평가)

## 품질 평가 기준

| 기준 | 가중치 | 설명 |
|------|--------|------|
| 조건 사용 완전성 | 25% | 문제의 모든 조건 활용 |
| 논리 전개 명확성 | 25% | STEP 간 자연스러운 연결 |
| 재현 가능성 | 30% | 학생이 따라할 수 있는 풀이 |
| 수식 표현 정확성 | 20% | LaTeX 문법 정확성 |

## 라이선스

This project is for educational purposes.