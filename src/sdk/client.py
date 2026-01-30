"""
ComfyUI RAG SDK - Client
Sync & Async Client for RAG API
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import httpx

from .models import (
    SearchResponse,
    IngestResponse,
    DocSearchResponse,
    DocIngestResponse,
    PushResponse,
    OutboxItem,
    WorkflowInfo,
)

DEFAULT_TIMEOUT = 30.0


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items() if v is not None}
    return value


def _clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = {k: v for k, v in payload.items() if v is not None}
    return _serialize_value(base)


class ComfyRagClient:
    def __init__(
        self,
        base_url: str = "http://localhost:7001",
        timeout: float = DEFAULT_TIMEOUT,
        raise_for_status: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.raise_for_status = raise_for_status
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _post(
        self,
        path: str,
        json_body: dict,
        model_cls=None,
        raise_for_status: Optional[bool] = None,
    ):
        if raise_for_status is None:
            raise_for_status = self.raise_for_status
        payload = _clean_payload(json_body)
        resp = self.client.post(path, json=payload)
        if raise_for_status:
            resp.raise_for_status()
        if model_cls and resp.status_code == 200:
            return model_cls(**resp.json())
        return resp.json()

    def _get(
        self,
        path: str,
        params: dict = None,
        model_cls=None,
        list_model: bool = False,
        raise_for_status: Optional[bool] = None,
    ):
        if raise_for_status is None:
            raise_for_status = self.raise_for_status
        clean_params = _clean_payload(params or {})
        resp = self.client.get(path, params=clean_params)
        if raise_for_status:
            resp.raise_for_status()

        data = resp.json()
        if model_cls:
            if list_model and isinstance(data, list):
                return [model_cls(**item) for item in data]
            if not list_model:
                return model_cls(**data)
        return data

    # --- Workflow ---
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        diversify: bool = False,
        mmr_lambda: float = 0.6,
        mmr_pool: int = 80,
        model: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[Union[datetime, str]] = None,
        date_to: Optional[Union[datetime, str]] = None,
        include_workflow_json: bool = False,
        **kwargs,
    ) -> SearchResponse:
        payload = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "diversify": diversify,
            "mmr_lambda": mmr_lambda,
            "mmr_pool": mmr_pool,
            "model": model,
            "status": status,
            "date_from": date_from,
            "date_to": date_to,
            "include_workflow_json": include_workflow_json,
            **kwargs,
        }
        return self._post("/search", payload, SearchResponse)

    def workflow_ingest(
        self,
        client_run_id: str,
        workflow_json: dict,
        prompt_positive: Optional[str] = None,
        prompt_negative: Optional[str] = None,
        models: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: str = "success",
        **kwargs,
    ) -> IngestResponse:
        payload = {
            "client_run_id": client_run_id,
            "workflow_json": workflow_json,
            "prompt_positive": prompt_positive,
            "prompt_negative": prompt_negative,
            "models": models or [],
            "tags": tags or [],
            "status": status,
            **kwargs,
        }
        return self._post("/ingest/workflow", payload, IngestResponse)

    # --- Knowledge ---
    def knowledge_search(
        self,
        query: str,
        type: str = None,
        tags: List[str] = None,
        limit: int = 10,
        diversify: bool = False,
        mmr_lambda: float = 0.6,
        mmr_pool: int = 80,
        offset: int = 0,
        date_from: Optional[Union[datetime, str]] = None,
        date_to: Optional[Union[datetime, str]] = None,
        **kwargs,
    ) -> DocSearchResponse:
        payload = {
            "query": query,
            "type": type,
            "tags": tags or [],
            "limit": limit,
            "offset": offset,
            "diversify": diversify,
            "mmr_lambda": mmr_lambda,
            "mmr_pool": mmr_pool,
            "date_from": date_from,
            "date_to": date_to,
            **kwargs,
        }
        return self._post("/knowledge/search", payload, DocSearchResponse)

    def knowledge_ingest(
        self,
        type: str,
        title: str,
        content: str,
        source_path: str = "sdk",
        chunk_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> DocIngestResponse:
        payload = {
            "type": type,
            "title": title,
            "content": content,
            "source_path": source_path,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "tags": tags or [],
            **kwargs,
        }
        return self._post("/knowledge/ingest", payload, DocIngestResponse)

    # --- Recall ---
    def recall_push(
        self,
        workflow_id: str,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        prompt_positive: Optional[str] = None,
        prompt_negative: Optional[str] = None,
        target_host: Optional[str] = None,
        **kwargs,
    ) -> PushResponse:
        payload = {
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "prompt_positive": prompt_positive,
            "prompt_negative": prompt_negative,
            "target_host": target_host,
            **kwargs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._post(f"/recall/{workflow_id}/push", payload, PushResponse)

    def recall_outbox(self, status: str = None, limit: int = 50) -> List[OutboxItem]:
        params = {"limit": limit}
        if status:
            params["status"] = status
        return self._get("/recall/outbox/list", params, OutboxItem, list_model=True)

    def get_outbox(self, status: str = None, limit: int = 50) -> List[OutboxItem]:
        return self.recall_outbox(status=status, limit=limit)

    def recall_info(self, workflow_id: str) -> WorkflowInfo:
        return self._get(f"/recall/{workflow_id}", model_cls=WorkflowInfo)


class AsyncComfyRagClient:
    def __init__(
        self,
        base_url: str = "http://localhost:7001",
        timeout: float = DEFAULT_TIMEOUT,
        raise_for_status: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.raise_for_status = raise_for_status
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _post(
        self,
        path: str,
        json_body: dict,
        model_cls=None,
        raise_for_status: Optional[bool] = None,
    ):
        if raise_for_status is None:
            raise_for_status = self.raise_for_status
        payload = _clean_payload(json_body)
        resp = await self.client.post(path, json=payload)
        if raise_for_status:
            resp.raise_for_status()
        if model_cls and resp.status_code == 200:
            return model_cls(**resp.json())
        return resp.json()

    async def _get(
        self,
        path: str,
        params: dict = None,
        model_cls=None,
        list_model: bool = False,
        raise_for_status: Optional[bool] = None,
    ):
        if raise_for_status is None:
            raise_for_status = self.raise_for_status
        clean_params = _clean_payload(params or {})
        resp = await self.client.get(path, params=clean_params)
        if raise_for_status:
            resp.raise_for_status()

        data = resp.json()
        if model_cls:
            if list_model and isinstance(data, list):
                return [model_cls(**item) for item in data]
            if not list_model:
                return model_cls(**data)
        return data

    # --- Workflow ---
    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        diversify: bool = False,
        mmr_lambda: float = 0.6,
        mmr_pool: int = 80,
        model: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[Union[datetime, str]] = None,
        date_to: Optional[Union[datetime, str]] = None,
        include_workflow_json: bool = False,
        **kwargs,
    ) -> SearchResponse:
        payload = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "diversify": diversify,
            "mmr_lambda": mmr_lambda,
            "mmr_pool": mmr_pool,
            "model": model,
            "status": status,
            "date_from": date_from,
            "date_to": date_to,
            "include_workflow_json": include_workflow_json,
            **kwargs,
        }
        return await self._post("/search", payload, SearchResponse)

    async def workflow_ingest(
        self,
        client_run_id: str,
        workflow_json: dict,
        prompt_positive: Optional[str] = None,
        prompt_negative: Optional[str] = None,
        models: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: str = "success",
        **kwargs,
    ) -> IngestResponse:
        payload = {
            "client_run_id": client_run_id,
            "workflow_json": workflow_json,
            "prompt_positive": prompt_positive,
            "prompt_negative": prompt_negative,
            "models": models or [],
            "tags": tags or [],
            "status": status,
            **kwargs,
        }
        return await self._post("/ingest/workflow", payload, IngestResponse)

    # --- Knowledge ---
    async def knowledge_search(
        self,
        query: str,
        type: str = None,
        tags: List[str] = None,
        limit: int = 10,
        diversify: bool = False,
        mmr_lambda: float = 0.6,
        mmr_pool: int = 80,
        offset: int = 0,
        date_from: Optional[Union[datetime, str]] = None,
        date_to: Optional[Union[datetime, str]] = None,
        **kwargs,
    ) -> DocSearchResponse:
        payload = {
            "query": query,
            "type": type,
            "tags": tags or [],
            "limit": limit,
            "offset": offset,
            "diversify": diversify,
            "mmr_lambda": mmr_lambda,
            "mmr_pool": mmr_pool,
            "date_from": date_from,
            "date_to": date_to,
            **kwargs,
        }
        return await self._post("/knowledge/search", payload, DocSearchResponse)

    async def knowledge_ingest(
        self,
        type: str,
        title: str,
        content: str,
        source_path: str = "sdk",
        chunk_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> DocIngestResponse:
        payload = {
            "type": type,
            "title": title,
            "content": content,
            "source_path": source_path,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "tags": tags or [],
            **kwargs,
        }
        return await self._post("/knowledge/ingest", payload, DocIngestResponse)

    # --- Recall ---
    async def recall_push(
        self,
        workflow_id: str,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        prompt_positive: Optional[str] = None,
        prompt_negative: Optional[str] = None,
        target_host: Optional[str] = None,
        **kwargs,
    ) -> PushResponse:
        payload = {
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "prompt_positive": prompt_positive,
            "prompt_negative": prompt_negative,
            "target_host": target_host,
            **kwargs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self._post(f"/recall/{workflow_id}/push", payload, PushResponse)

    async def recall_outbox(self, status: str = None, limit: int = 50) -> List[OutboxItem]:
        params = {"limit": limit}
        if status:
            params["status"] = status
        return await self._get("/recall/outbox/list", params, OutboxItem, list_model=True)

    async def get_outbox(self, status: str = None, limit: int = 50) -> List[OutboxItem]:
        return await self.recall_outbox(status=status, limit=limit)

    async def recall_info(self, workflow_id: str) -> WorkflowInfo:
        return await self._get(f"/recall/{workflow_id}", model_cls=WorkflowInfo)
