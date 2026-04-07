# 수능 수학 문제 풀이 분석 플랫폼

수능 수학 문제를 이미지 또는 텍스트로 입력하면 난이도 분류, 풀이, 답, 개념, 흐름을 반환하는 프로젝트입니다.  
구성은 `FastAPI 백엔드`와 `React + Vite 프론트엔드`로 나뉘며, 문서는 이 README 하나만 유지합니다.

## 구조

```text
.
├── requirements.txt              # 루트 Python 의존성 파일
├── capstone4-1-backend/          # FastAPI + LangGraph
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example
└── capstone4-1-frontend/         # React + Vite + Nginx
    ├── src/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── nginx.conf
    └── .env.example
```

## 기술 스택

- Backend: Python, FastAPI, LangGraph, OpenAI API, Uvicorn
- Frontend: React, TypeScript, Vite, Tailwind CSS, Radix UI
- Infra: Docker, Docker Compose, Nginx

## 환경 관리 원칙

- Python 가상환경은 루트 `.venv` 하나만 사용합니다.
- 백엔드는 루트 `requirements.txt`를 사용합니다.
- 프론트는 Node.js 프로젝트라 `package.json`과 `node_modules`로 의존성을 관리합니다.

## 로컬 실행

### 1. 백엔드

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp capstone4-1-backend/.env.example capstone4-1-backend/.env
cd capstone4-1-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

백엔드 환경변수:

```env
OPENAI_API_KEY=your_openai_api_key
APP_ENV=local
LOG_LEVEL=INFO
MAX_IMAGE_SIZE_MB=10
```

### 2. 프론트엔드

```bash
cd capstone4-1-frontend
npm install
cp .env.example .env
npm run dev
```

프론트 환경변수:

```env
VITE_API_URL=http://localhost:8000
```

개발 접속 주소:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## Docker 실행

루트에서 한 번에 실행할 수 있습니다.

```bash
cp capstone4-1-backend/.env.example capstone4-1-backend/.env
docker compose up --build
```

이 명령으로 백엔드와 프론트가 각각 다른 컨테이너로 함께 실행됩니다.

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

개별 실행도 가능합니다.

- 백엔드만: [capstone4-1-backend/docker-compose.yml](/Users/limgiheon/Desktop/캡스톤/capstone4-1/캡스톤4-1/capstone4-1-backend/docker-compose.yml)
- 프론트만: [capstone4-1-frontend/docker-compose.yml](/Users/limgiheon/Desktop/캡스톤/capstone4-1/캡스톤4-1/capstone4-1-frontend/docker-compose.yml)

프론트 Docker는 Vite 결과물을 Nginx로 서빙합니다.
[capstone4-1-frontend/nginx.conf](/Users/limgiheon/Desktop/캡스톤/capstone4-1/캡스톤4-1/capstone4-1-frontend/nginx.conf)는 정적 파일 서빙, SPA 라우팅 fallback, `/healthz` 헬스체크를 담당합니다.

## API

- `GET /api/v1/health`
- `POST /api/v1/solve`

`POST /api/v1/solve`는 `multipart/form-data`를 받습니다.

- `question_text: str | None`
- `image: UploadFile | None`

둘 중 하나는 반드시 있어야 합니다.

## 메모

- 프론트 빌드 시 `500kB` 번들 경고가 나올 수 있습니다.
  의미는 빌드는 성공했지만 초기 JS 크기가 커서 최적화 여지가 있다는 뜻입니다.
- 루트 `.gitignore` 하나로 백엔드와 프론트의 산출물, 가상환경, 캐시를 함께 관리합니다.
