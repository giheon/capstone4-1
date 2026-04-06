# 수능 수학 문제 풀이 분석 플랫폼

수능 수학 문제를 이미지로 업로드하면 AI가 문제를 분석하고 단계별 풀이와 관련 개념을 제공하는 플랫폼입니다.

## 프로젝트 구조

```
App/
├── capstone4-1-backend/     # FastAPI 백엔드
│   ├── app/
│   │   ├── api/             # API 라우트
│   │   ├── core/            # 설정, 로깅
│   │   ├── domain/          # 비즈니스 로직 (개념 카탈로그 등)
│   │   ├── llm/             # LLM 클라이언트
│   │   ├── schemas/         # Pydantic 스키마
│   │   ├── workflows/       # LangGraph 워크플로우
│   │   └── main.py          # FastAPI 앱 진입점
│   ├── tests/               # 테스트
│   ├── requirements.txt     # Python 의존성
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── capstone4-1-frontend/    # React + Vite 프론트엔드
    ├── src/
    │   ├── api/             # API 호출
    │   ├── components/      # UI 컴포넌트
    │   ├── context/         # React Context
    │   ├── pages/           # 페이지 컴포넌트
    │   └── routes.tsx       # 라우팅
    ├── package.json
    └── vite.config.ts
```

## 기술 스택

### Backend
- **Python 3.11+**
- **FastAPI** - 웹 프레임워크
- **LangGraph** - LLM 워크플로우
- **OpenAI API** - GPT 모델 활용
- **Uvicorn** - ASGI 서버

### Frontend
- **React 18** + **TypeScript**
- **Vite** - 빌드 도구
- **Tailwind CSS** - 스타일링
- **Radix UI** - UI 컴포넌트
- **React Router** - 라우팅
- **KaTeX** - 수학 수식 렌더링

---

## 실행 방법

### 사전 요구사항
- Python 3.11 이상
- Node.js 18 이상
- OpenAI API Key

---

### 1. Backend 실행

```bash
# 백엔드 디렉토리로 이동
cd capstone4-1-backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 설정
```

**.env 파일 설정:**
```env
OPENAI_API_KEY=your_openai_api_key_here
APP_ENV=local
LOG_LEVEL=INFO
MAX_IMAGE_SIZE_MB=10
```

```bash
# 서버 실행 (http://localhost:8000)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker로 실행 (선택사항)
```bash
cd capstone4-1-backend
docker-compose up --build
```

---

### 2. Frontend 실행

```bash
# 프론트엔드 디렉토리로 이동
cd capstone4-1-frontend

# 의존성 설치
npm install

# 환경변수 설정
cp .env.example .env
# .env 파일 확인 (기본값: http://localhost:8000)
```

**.env 파일 설정:**
```env
VITE_API_URL=http://localhost:8000
```

```bash
# 개발 서버 실행 (http://localhost:3000)
npm run dev

# 프로덕션 빌드
npm run build
```

---

### 3. 전체 실행 순서

1. **터미널 1 - Backend:**
   ```bash
   cd capstone4-1-backend
   source venv/bin/activate
   uvicorn app.main:app --reload --port 8000
   ```

2. **터미널 2 - Frontend:**
   ```bash
   cd capstone4-1-frontend
   npm run dev
   ```

3. 브라우저에서 `http://localhost:3000` 접속

---

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/health` | 서버 상태 확인 |
| POST | `/api/v1/solve` | 수학 문제 풀이 요청 |

---

## 주요 기능

1. **문제 이미지 업로드** - 수능 수학 문제 이미지를 업로드
2. **AI 문제 분석** - GPT 모델이 문제를 인식하고 분석
3. **단계별 풀이 제공** - 상세한 풀이 과정 제시
4. **관련 개념 연결** - 문제와 관련된 수학 개념 제공
5. **수식 렌더링** - KaTeX를 활용한 수학 수식 표시

---

## 환경변수 요약

### Backend (.env)
| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - |
| `APP_ENV` | 실행 환경 | `local` |
| `LOG_LEVEL` | 로그 레벨 | `INFO` |
| `MAX_IMAGE_SIZE_MB` | 최대 이미지 크기 | `10` |

### Frontend (.env)
| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `VITE_API_URL` | 백엔드 API URL | `http://localhost:8000` |

---

## 라이선스

이 프로젝트는 학술 목적으로 개발되었습니다.
