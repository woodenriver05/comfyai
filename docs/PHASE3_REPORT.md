# Phase 3: LangGraph Multi-Agent Pipeline Report

**Date**: 2026-01-29
**Status**: E2E 통과 (19/19)

---

## Architecture

```
START -> supervisor -> planner -> supervisor -> retriever -> grade_results
      -> supervisor -> architect -> supervisor -> validator -> supervisor
      -> human_review -> supervisor -> executor -> respond -> END
```

### Nodes (8)

| Node | Layer | Role |
|------|-------|------|
| **supervisor** | L1 | Intent 분류 + 상태 기반 라우팅 |
| **planner** | L2 | Creative/Technical Plan + 영어 검색 쿼리 생성 |
| **retriever** | L2 | search_workflows 호출 (model 필터, 재시도 지원) |
| **grade_results** | L2 | signals 기반 품질 판정 (good/fair/poor) |
| **architect** | L2 | 워크플로우 JSON 경로 정규화 + 모델 매핑 |
| **validator** | L3 | 규칙 기반 JSON 구조 검증 |
| **human_review** | L3 | 실행 전 최종 확인 요약 (auto-approve) |
| **executor** | L2 | 기존 ReAct 에이전트 실행 |

### Routing Table

| Intent | Flow |
|--------|------|
| `generate` | planner -> retriever -> grade -> architect -> validator -> human_review -> executor |
| `search` | executor (직행) |
| `knowledge` | executor (직행) |
| `execute` | executor (직행) |

### Circuit Breaker
- `grade == poor` && `retries < 2` -> retriever 재시도 (model 필터 제거)
- `retries >= 2` -> architect로 진행 (best effort)

---

## E2E Test Results

### Scenario 1: Generate (Full Pipeline)
- **Query**: "비 오는 사이버펑크 도시의 네온 거리, wan2.2로 1080p 영상 만들어줘"
- **Intent**: `generate`
- **Planner output**: mood=moody/dark/cinematic, style=cyberpunk, model=wan2.2, res=1920x1080, pipe=T2V
- **Search query**: "cyberpunk rainy city neon streets T2V wan2.2 video"
- **Retriever**: 첫 시도 0건 (model 필터) -> 재시도 5건, top_score=0.434
- **Grade**: good
- **Steps**: 17, **Time**: 33s

### Scenario 2: Search (Direct)
- **Query**: "wan2.2 비디오 워크플로우 찾아줘"
- **Intent**: `search` -> executor 직행
- **Steps**: 3, **Time**: 14s

### Scenario 3: Knowledge (Direct)
- **Query**: "ComfyUI에서 GGUF 모델을 로드하는 방법 알려줘"
- **Intent**: `knowledge` -> executor 직행
- **Steps**: 3, **Time**: 20s

---

## Key Files

```
src/agent/
  state.py              # AgentState TypedDict (공유 상태)
  prompts.py            # 한국어 프롬프트 상수
  supervisor_graph.py   # 멀티에이전트 그래프 조립 + 노드 함수
  architect.py          # 경로 정규화 + 모델 매핑
  graph.py              # LangGraph Studio 진입점
  agent.py              # 기존 ReAct 에이전트 (미수정)
  tools.py              # LangChain 도구 5종 (미수정)
langgraph.json          # LangGraph Studio 설정
test_e2e_rag.py         # E2E 테스트 (3 시나리오)
test_graph_stream.py    # 스트리밍 테스트
```

## Issues Resolved

1. **`load_dotenv()` override 문제**: 쉘 환경변수에 만료된 GOOGLE_API_KEY가 있으면 `.env`의 새 키가 무시됨. `load_dotenv(override=True)`로 해결.

2. **`google-genai` SDK contents 필수**: `with_structured_output()` 사용 시 system 메시지만으로는 `ValueError: contents are required` 발생. `("human", "{query}")` 메시지 추가로 해결.

3. **Architect graceful degradation**: `workflow_json`이 없는 검색 결과에 대해 skip 처리 + 변경 로그 기록.

---

## Stack

- **LLM**: Gemini 2.5 Pro (langchain-google-genai 4.2.0)
- **Graph**: LangGraph 0.6.11
- **RAG DB**: PostgreSQL + pgvector (106 workflows)
- **Embedding**: all-MiniLM-L6-v2 (384d)
- **Studio**: LangGraph Studio via `langgraph dev`
