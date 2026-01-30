"""
Phase 3.5: ComfyUI RAG Agent - LangChain Tools (Structured JSON)

모든 도구는 JSON 문자열을 반환한다.
SDK의 기존 Pydantic 모델을 그대로 활용하고, 판단 힌트(signals)만 추가.
"""
import json
import os
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain.tools import tool

from ..sdk.client import ComfyRagClient

# --- Config ---

SCORE_THRESHOLD = 0.25
GAP_THRESHOLD = 0.05

_client: Optional[ComfyRagClient] = None
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:7001")


# --- Client Management ---

def _get_client() -> ComfyRagClient:
    global _client
    if _client is None:
        _client = ComfyRagClient(base_url=RAG_API_URL)
    return _client


def set_client(client: ComfyRagClient):
    """테스트 등에서 클라이언트를 직접 주입할 때 사용"""
    global _client
    _client = client


# --- JSON Helpers ---

def _dump(obj):
    """Pydantic 모델 또는 리스트를 JSON-serializable dict로 변환"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_dump(item) for item in obj]
    return obj


def _json(data: dict) -> str:
    """dict를 JSON 문자열로 직렬화 (datetime 처리 포함)"""
    return json.dumps(data, ensure_ascii=False, default=str)


def _is_retryable(exc: Exception) -> bool:
    """네트워크/타임아웃 등 재시도 가능한 에러인지 판단"""
    retryable_types = (ConnectionError, TimeoutError, OSError)
    msg = str(exc).lower()
    return isinstance(exc, retryable_types) or "timeout" in msg or "connection" in msg


def _json_error(code: str, message: str, retryable: bool = False) -> str:
    """구조화된 에러 JSON 반환"""
    return _json({"error": {"code": code, "message": message, "retryable": retryable}})


def _preview_overrides(info, prompt_positive: Optional[str], prompt_negative: Optional[str], seed: Optional[int]) -> dict:
    """WorkflowInfo 기반으로 오버라이드 미리보기 생성 (original vs override diff)"""
    def _field(original, override):
        return {"original": original, "override": override, "changed": override is not None and override != original}

    orig_pos = getattr(info, "last_prompt_positive", None)
    orig_neg = getattr(info, "last_prompt_negative", None)
    orig_seed = getattr(info, "last_seed", None)

    return {
        "workflow_id": getattr(info, "workflow_id", None),
        "prompt_positive": _field(orig_pos, prompt_positive),
        "prompt_negative": _field(orig_neg, prompt_negative),
        "seed": _field(orig_seed, seed),
    }


def _compute_signals(results: list) -> dict:
    """검색 결과로부터 판단 힌트(signals)를 계산"""
    no_results = len(results) == 0
    top_score = float(results[0]["score"]) if results else None
    gap = None
    if len(results) >= 2:
        gap = round(float(results[0]["score"]) - float(results[1]["score"]), 4)
    return {
        "no_results": no_results,
        "low_confidence": no_results or (top_score is not None and top_score < SCORE_THRESHOLD),
        "ambiguous": (not no_results) and gap is not None and gap < GAP_THRESHOLD,
        "top_score": round(top_score, 4) if top_score is not None else None,
        "score_gap": gap,
    }


# --- Pydantic Schemas for Tool Arguments ---

class SearchWorkflowsArgs(BaseModel):
    query: str = Field(description="Natural language query to search workflows.")
    limit: int = Field(default=5, description="Max results to return.")
    diversify: bool = Field(default=False, description="Use MMR reranking for diversity.")
    mmr_lambda: float = Field(default=0.6, description="MMR tradeoff (0=diversity, 1=relevance).")
    mmr_pool: int = Field(default=80, description="Candidate pool size for MMR.")
    model: Optional[str] = Field(default=None, description="Filter by model name (e.g. 'wan2.2').")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags (ANY match).")
    status: Optional[str] = Field(default=None, description="Filter by status (e.g. 'success').")
    date_from: Optional[str] = Field(default=None, description="Filter start date (YYYY-MM-DD).")
    date_to: Optional[str] = Field(default=None, description="Filter end date (YYYY-MM-DD).")


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(description="Natural language query to search knowledge docs.")
    limit: int = Field(default=5, description="Max results to return.")
    diversify: bool = Field(default=False, description="Use MMR reranking for diversity.")
    mmr_lambda: float = Field(default=0.6, description="MMR tradeoff (0=diversity, 1=relevance).")
    mmr_pool: int = Field(default=80, description="Candidate pool size for MMR.")
    type: Optional[str] = Field(default=None, description="Filter by doc type.")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags.")


class GetWorkflowInfoArgs(BaseModel):
    workflow_id: str = Field(description="The UUID of the workflow.")


class PushWorkflowArgs(BaseModel):
    workflow_id: str = Field(description="The UUID of the workflow to push.")
    prompt_positive: Optional[str] = Field(default=None, description="Override positive prompt.")
    prompt_negative: Optional[str] = Field(default=None, description="Override negative prompt.")
    seed: Optional[int] = Field(default=None, description="Specific seed for generation.")
    dry_run: bool = Field(default=True, description="If true, validate only without actual push. Always use true first.")


class CheckOutboxArgs(BaseModel):
    status: Optional[str] = Field(default="pending", description="Filter: 'pending', 'sent', or 'failed'.")
    limit: int = Field(default=10, description="Max items to return.")


# --- LangChain Tools ---

@tool(args_schema=SearchWorkflowsArgs)
def search_workflows(
    query: str,
    limit: int = 5,
    diversify: bool = False,
    mmr_lambda: float = 0.6,
    mmr_pool: int = 80,
    model: Optional[str] = None,
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Searches for ComfyUI workflows. Returns JSON with results, meta, signals, and error.
    Use signals.low_confidence and signals.ambiguous to decide next action."""
    try:
        client = _get_client()
        effective_limit = limit * 3 if tags else limit
        response = client.search(
            query=query,
            limit=effective_limit,
            diversify=diversify,
            mmr_lambda=mmr_lambda,
            mmr_pool=mmr_pool,
            model=model,
            status=status,
            date_from=date_from,
            date_to=date_to,
            include_workflow_json=False,
        )

        results = [_dump(r) for r in response.results]
        if tags:
            tagset = {t.lower() for t in tags}
            results = [r for r in results if any(str(t).lower() in tagset for t in (r.get("tags") or []))]
        results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        results = results[:limit]

        return _json({
            "results": results,
            "meta": {
                "query": query,
                "total_raw": getattr(response, "total", len(results)),
                "total": len(results),
                "filters": {
                    k: v for k, v in {
                        "model": model, "tags": tags, "status": status,
                        "date_from": date_from, "date_to": date_to,
                        "diversify": diversify, "mmr_lambda": mmr_lambda,
                        "mmr_pool": mmr_pool,
                    }.items() if v is not None
                },
            },
            "signals": _compute_signals(results),
            "error": None,
        })
    except Exception as e:
        return _json_error("SEARCH_FAILED", str(e), retryable=_is_retryable(e))


@tool(args_schema=SearchKnowledgeArgs)
def search_knowledge(
    query: str,
    limit: int = 5,
    diversify: bool = False,
    mmr_lambda: float = 0.6,
    mmr_pool: int = 80,
    type: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Searches the knowledge base (docs, worklogs, architecture). Returns JSON with results, meta, signals, error."""
    try:
        client = _get_client()
        response = client.knowledge_search(
            query=query,
            limit=limit,
            diversify=diversify,
            mmr_lambda=mmr_lambda,
            mmr_pool=mmr_pool,
            type=type,
            tags=tags or [],
        )

        results = [_dump(r) for r in response.results]
        results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

        return _json({
            "results": results,
            "meta": {
                "query": query,
                "total_raw": getattr(response, "total", len(results)),
                "total": len(results),
                "filters": {
                    k: v for k, v in {"type": type, "tags": tags}.items() if v is not None
                },
            },
            "signals": _compute_signals(results),
            "error": None,
        })
    except Exception as e:
        return _json_error("KNOWLEDGE_SEARCH_FAILED", str(e), retryable=_is_retryable(e))


@tool(args_schema=GetWorkflowInfoArgs)
def get_workflow_info(workflow_id: str) -> str:
    """Gets detailed info about a specific workflow. Returns JSON."""
    try:
        client = _get_client()
        info = client.recall_info(workflow_id=workflow_id)
        return _json({"workflow": _dump(info), "error": None})
    except Exception as e:
        return _json_error("GET_INFO_FAILED", str(e), retryable=_is_retryable(e))


@tool(args_schema=PushWorkflowArgs)
def push_workflow(
    workflow_id: str,
    prompt_positive: Optional[str] = None,
    prompt_negative: Optional[str] = None,
    seed: Optional[int] = None,
    dry_run: bool = True,
) -> str:
    """Pushes a workflow to the execution outbox. Always use dry_run=true first.
    Only set dry_run=false after user explicitly confirms execution."""
    try:
        client = _get_client()

        if dry_run:
            # dry_run: 실제 push 없이 워크플로우 정보 + 오버라이드 미리보기 반환
            info = client.recall_info(workflow_id=workflow_id)
            return _json({
                "ok": True,
                "dry_run": True,
                "workflow": _dump(info),
                "preview": _preview_overrides(info, prompt_positive, prompt_negative, seed),
                "message": "Dry run complete. Confirm to execute with dry_run=false.",
                "error": None,
            })

        # 실제 push
        response = client.recall_push(
            workflow_id=workflow_id,
            prompt_positive=prompt_positive,
            prompt_negative=prompt_negative,
            seed=seed,
        )
        return _json({
            "ok": True,
            "dry_run": False,
            "outbox_id": response.outbox_id,
            "status": response.status,
            "message": response.message,
            "error": None,
        })
    except Exception as e:
        return _json_error("PUSH_FAILED", str(e), retryable=_is_retryable(e))


@tool(args_schema=CheckOutboxArgs)
def check_outbox(status: Optional[str] = "pending", limit: int = 10) -> str:
    """Checks execution outbox status. Returns JSON list of items."""
    try:
        client = _get_client()
        items = client.recall_outbox(status=status, limit=limit)
        return _json({
            "items": [_dump(item) for item in items],
            "meta": {"status_filter": status, "count": len(items)},
            "error": None,
        })
    except Exception as e:
        return _json_error("OUTBOX_CHECK_FAILED", str(e), retryable=_is_retryable(e))


# --- Tool Registry ---

def get_tools() -> list:
    """에이전트에 바인딩할 모든 도구 목록 반환"""
    return [
        search_workflows,
        search_knowledge,
        get_workflow_info,
        push_workflow,
        check_outbox,
    ]
