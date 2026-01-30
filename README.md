# ComfyUI Memory RAG v1

AI 협업을 위한 ComfyUI 워크플로우 기억 시스템

## Quick Start

```bash
# 1. 환경 설정
cp .env.example .env
vi .env  # DATABASE_URL 확인

# 2. DB 시작 (Postgres + pgvector)
docker compose up -d

# 3. 의존성 설치 (uv 권장)
uv add fastapi "uvicorn[standard]" pydantic python-dotenv \
       asyncpg pgvector numpy sentence-transformers
uv add --dev pytest pytest-asyncio httpx ruff

# 또는 pip
pip install -r requirements.txt

# 4. API 서버 실행
uvicorn src.api.main:app --host 0.0.0.0 --port 7001 --reload
```

## API Endpoints

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/health` | GET | 헬스체크 |
| `/ingest/workflow` | POST | 워크플로우 저장 |
| `/search` | POST | 자연어 검색 |
| `/search/quick?q=` | GET | 빠른 검색 |
| `/recall/{id}` | GET | 워크플로우 정보 |
| `/recall/{id}/push` | POST | Outbox에 추가 (PC 대기열) |
| `/recall/{id}/json` | GET | 원본 JSON |
| `/recall/outbox/list` | GET | Outbox 대기열 조회 |

## Mac 단독 모드

PC 연결 전까지는 `/recall/{id}/push`가 `recall_outbox` 테이블에 적재만 합니다.
PC 연결 후에는 별도 sender 데몬이 outbox를 폴링하여 송신합니다. (추후 구현)

## Smoke Test (Mac 단독)

```bash
# 1. Health check
curl http://localhost:7001/health

# 2. Ingest (샘플)
curl -X POST http://localhost:7001/ingest/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "client_run_id": "test-001",
    "workflow_json": {"3": {"class_type": "KSampler", "inputs": {"seed": 42}}},
    "prompt_positive": "anime girl, high quality",
    "models": ["animagine-xl"]
  }'

# 3. Search
curl -X POST http://localhost:7001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "anime"}'

# 4. Push (Outbox 적재)
curl -X POST http://localhost:7001/recall/{workflow_id}/push

# 5. Outbox 확인
curl http://localhost:7001/recall/outbox/list
```

## 테이블 구조

```
workflows          - 워크플로우 템플릿 (중복 방지)
generations        - 각 실행 기록
models             - 모델/LoRA 레지스트리
generation_models  - 다대다 연결
recall_outbox      - PC 재실행 대기열
```

## Architecture

```
Mac Mini (Brain)              PC (Engine) - 추후 연결
┌────────────────────┐        ┌─────────────┐
│ Memory RAG API     │   →    │ ComfyUI     │
│ - Postgres+pgvec   │        │ - Queue API │
│ - FastAPI          │        │ - 5090 GPU  │
│ - recall_outbox    │        │             │
└────────────────────┘        └─────────────┘
```

## 환경 변수

```bash
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/comfyui_rag
EMBEDDING_MODEL=all-MiniLM-L6-v2
API_PORT=7001                              # macOS AirPlay가 7000 점유
COMFYUI_API_URL=http://192.168.1.100:8188  # PC 주소 (sender에서 사용, 추후)
```
