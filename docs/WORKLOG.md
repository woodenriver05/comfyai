# 🏭 ComfyUI Agent RAG - Project Journal

**Technical Director**: Gemini Factory
**Platform**: Mac Mini M4 Pro (Brain) ↔ High-End PC (Engine, Planned)
**Status**: Phase 3 Complete (v1.5 Agent Edition)

---

## 🏛️ Current Architecture (v1.5)

### 1. System Overview
에이전트의 **기억(Memory)**과 **지식(Knowledge)**을 관리하고, **자연어로 워크플로우를 실행**하는 통합 RAG 시스템입니다.

*   **API Server**: FastAPI (Port 7001)
*   **Database**: PostgreSQL 16 + `pgvector` (Docker)
*   **Embedding**: `all-MiniLM-L6-v2` (384 dim)
*   **Client SDK**: Python 전용 SDK (`src/sdk`) 제공 (Sync/Async 지원, v1.1 최적화 완료)
*   **Agent**: LangGraph ReAct 에이전트 (`src/agent`) + Gemini 2.5 Pro
*   **Sender Daemon**: Outbox 폴링 및 ComfyUI PC 전송 (`src/daemon/sender.py`)

### 2. Database Schema (Core Tables)
*   **`workflows`**: ComfyUI 워크플로우 템플릿. (절차적 기억)
*   **`generations`**: 실행 이력 및 파라미터. (경험적 기억)
*   **`documents`**: 기술 문서, 작업 일지, 아키텍처 리뷰 등. (의미적 지식)
*   **`recall_outbox`**: PC 전송 대기열.

### 3. Key Logic
*   **Knowledge Base Extension**: 워크플로우뿐만 아니라 마크다운 문서를 청킹하여 저장/검색 가능.
*   **MMR Diversity Reranking**: 검색 결과의 다양성을 확보하기 위해 MMR(Maximal Marginal Relevance) 엔진 장착.
*   **Robust SDK**: 타입 안전성(Pydantic)과 명시적 파라미터를 지원하는 비동기 친화적 SDK.
*   **ReAct Agent**: LangGraph 기반 ReAct 패턴 에이전트. 5개 도구를 조합하여 자연어 요청 처리.

---

## 📅 Development Log

### [2026-01-28] Phase 3: Agent Prototype (v1.5)

#### ✅ Achievements (성과)
1.  **LangGraph ReAct 에이전트 구축**: `src/agent/` 모듈 신규 생성.
    *   5개 LangChain Tool: `search_workflows`, `search_knowledge`, `get_workflow_info`, `push_workflow`, `check_outbox`
    *   Gemini 2.5 Pro 기반 ReAct 패턴 에이전트 (`agent.py`)
    *   Pydantic `args_schema`로 도구 입력 타입 정의
2.  **시나리오 테스트 통과**:
    *   시나리오 1 (프롬프트 기반): "사이버펑크 영상" → 워크플로우 검색 → 추천 → Pass
    *   시나리오 2 (지식+워크플로우 복합): 아키텍처 설명 + 비디오 워크플로우 추천 → Pass
3.  **Phase 2.5 Sender Daemon 코드 완성**: `src/daemon/sender.py` 구현 완료 (E2E 테스트는 ComfyUI PC 연결 후 진행).
4.  **보안 강화**: 9개 에이전트용 ignore 파일 생성 (`.geminiignore`, `.claudeignore`, `.aiderignore`, `.cursorignore`, `.copilotignore`, `.gptignore`, `.codexignore`, `.kimiignore`, `.gitignore`).
5.  **환경 정리**: `.venv` (Python 3.12, uv) 기준 의존성 통일, `pyproject.toml` + `requirements.txt` 동기화.

#### 📝 Notes
*   **인증**: `GOOGLE_API_KEY` (AI Studio 발급, `AIzaSy...` 형식)를 `.env`에 보관. 에이전트 ignore 파일로 보호.
*   **Agent LLM**: Gemini 2.5 Pro, temperature 0.3. 한국어 응답.
*   **협업**: Claude Code (장인) + Gemini (제1조수) 공동 작업.

### [2026-01-28] Phase 1 & 2: Knowledge & Tooling (v1.4)

#### ✅ Achievements (성과)
1.  **Knowledge Base (KMS) 구축**: `documents` 테이블 및 `/knowledge` API 추가 및 안정화.
2.  **Bulk Data Ingest**: 105개 실전 워크플로우 학습 완료.
3.  **Search Quality Tuning**: MMR 리랭킹 엔진(`ranking.py`) 적용 및 다양성 검색 검증 완료.
4.  **Python SDK v1.1 업그레이드**:
    *   `recall_info()` 메서드 추가로 워크플로우 상세 정보 조회 기능 보강.
    *   `search()` 등 주요 메서드에 명시적 파라미터(`mmr_pool`, `diversify` 등) 노출하여 사용성 개선.
    *   `raise_for_status` 옵션으로 에러 처리 정책 유연화.
    *   Sync/Async 통합 테스트(`test_sdk.py`) 전 항목 Pass.

#### 📝 Notes
*   **Diversity Test**: `mmr_lambda=0.3` 설정 시 정확도보다 맥락의 다양성을 우선하여 풍부한 결과 반환 확인.
*   **Late-night Sprint**: Phase 2까지의 모든 핵심 기능을 성공적으로 마무리하고 Sender Daemon 단계로 진입 준비 완료.

---

## 🛠️ Operation & Usage Guide (사용법)

### 1. Data Ingest (데이터 주입)
*   **워크플로우**: `python scripts/bulk_ingest.py`
*   **기술 문서**: `python scripts/ingest_knowledge.py docs/WORKLOG.md`

### 2. SDK Usage (파이썬에서 사용)
```python
from src.sdk import ComfyRagClient

with ComfyRagClient("http://localhost:7001") as client:
    # 워크플로우 검색 (다양성 옵션)
    res = client.search("cyberpunk", diversify=True, mmr_lambda=0.3)
    # 특정 워크플로우 정보 조회
    info = client.recall_info(res.results[0].workflow_id)
    # Outbox 적재
    client.recall_push(workflow_id=info.workflow_id, seed=0)
```

### 3. Agent Usage (에이전트 사용)
```python
from src.agent import create_comfyui_agent

agent = create_comfyui_agent()
result = agent.invoke({"messages": [("user", "사이버펑크 영상 워크플로우 찾아줘")]})
```

### 4. Agent Test (에이전트 테스트)
```bash
# 도구 단위 테스트
.venv/bin/python scripts/test_agent.py tools

# 에이전트 시나리오 테스트
.venv/bin/python scripts/test_agent.py scenario1
.venv/bin/python scripts/test_agent.py scenario2
```

---

## 🗺️ Next Roadmap

### Phase 4: Production Integration
- [ ] ComfyUI PC 연결 후 Sender Daemon E2E 테스트.
- [ ] 에이전트 대화형 인터페이스 (Streamlit 또는 Gradio).
- [ ] 설정 통합 (`pydantic-settings` 기반 `src/config.py`).
- [ ] 에이전트 실행 이력 로깅 및 피드백 루프.
