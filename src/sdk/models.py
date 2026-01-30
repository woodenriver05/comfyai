"""
ComfyUI RAG SDK - Data Models
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

# --- Search ---
class WorkflowResult(BaseModel):
    workflow_id: str
    generation_id: Optional[str] = None
    score: float
    prompt_positive: Optional[str]
    prompt_negative: Optional[str]
    models: List[str]
    tags: List[str]
    success_count: int
    created_at: datetime
    result_paths: List[str] = []
    workflow_json: Optional[dict] = None

class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[WorkflowResult]

# --- Ingest ---
class IngestResponse(BaseModel):
    workflow_id: str
    generation_id: str
    workflow_hash: str
    status: str
    message: str

# --- Knowledge ---
class DocResult(BaseModel):
    id: str
    score: float
    type: str
    title: str
    content: str
    source_path: str
    chunk_index: int
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime

class DocSearchResponse(BaseModel):
    query: str
    total: int
    results: List[DocResult]

class DocIngestResponse(BaseModel):
    id: str
    doc_hash: str
    status: str
    message: str

# --- Recall ---
class PushResponse(BaseModel):
    success: bool
    workflow_id: str
    outbox_id: str
    status: str
    message: str
    workflow_json: Optional[dict] = None

class OutboxItem(BaseModel):
    id: str
    workflow_id: str
    status: str
    target_host: str
    overrides: dict
    attempts: int
    created_at: datetime
    last_attempt_at: Optional[datetime] = None

class WorkflowInfo(BaseModel):
    workflow_id: str
    name: Optional[str] = None
    models: List[str]
    tags: List[str]
    success_count: int
    created_at: datetime
    last_prompt_positive: Optional[str] = None
    last_prompt_negative: Optional[str] = None
    last_seed: Optional[int] = None
