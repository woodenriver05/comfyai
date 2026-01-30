"""
ComfyUI Memory RAG v1.1 - Knowledge Endpoint
POST /knowledge/ingest: 문서 청크 저장
POST /knowledge/search: 문서 검색
Prefix: /knowledge (to avoid conflict with /docs Swagger UI)
"""

import hashlib
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..database import get_pool
from ..embedder import generate_embedding_sync
from ..ranking import mmr_rerank

router = APIRouter()


# ============================================
# Request/Response Models
# ============================================

class DocIngestRequest(BaseModel):
    type: str = Field(..., description="worklog, architecture, etc.")
    title: str
    content: str
    source_path: str
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class DocIngestResponse(BaseModel):
    id: str
    doc_hash: str
    status: str
    message: str


class DocSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    
    # Filters
    type: Optional[str] = None
    tags: List[str] = Field(default_factory=list) # ANY match (&&)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)

    diversify: bool = False
    mmr_lambda: float = Field(0.6, ge=0.0, le=1.0)
    mmr_pool: int = Field(80, ge=10, le=120)


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


# ============================================
# Helpers
# ============================================

def compute_doc_hash(source_path: str, chunk_index: int, content: str) -> str:
    """Unique hash for deduplication"""
    raw = f"{source_path}:{chunk_index}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ============================================
# Endpoints
# ============================================

@router.post("/ingest", response_model=DocIngestResponse)
async def ingest_document(req: DocIngestRequest):
    """
    지식 문서 저장 (Upsert)
    """
    # 1. CPU Bound: Hash & Embedding
    doc_hash = compute_doc_hash(req.source_path, req.chunk_index, req.content)
    embedding = await run_in_threadpool(generate_embedding_sync, req.content)
    
    # 2. DB Transaction
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    doc_hash, type, title, content, source_path, chunk_index,
                    metadata, tags, embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)
                ON CONFLICT (doc_hash)
                DO UPDATE SET
                    type = EXCLUDED.type,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    tags = EXCLUDED.tags,
                    updated_at = NOW()
                RETURNING id, (xmax = 0) AS is_new
                """,
                doc_hash, req.type, req.title, req.content, req.source_path, req.chunk_index,
                json.dumps(req.metadata), req.tags, embedding
            )
            
            doc_id = str(row["id"])
            is_new = row["is_new"]

    return DocIngestResponse(
        id=doc_id,
        doc_hash=doc_hash,
        status="created" if is_new else "updated",
        message="Document saved successfully."
    )


@router.post("/search", response_model=DocSearchResponse)
async def search_documents(req: DocSearchRequest):
    """
    지식 문서 의미 검색
    """
    pool = await get_pool()
    
    # 1. Query Embedding
    query_vec = await run_in_threadpool(generate_embedding_sync, req.query)

    pool_limit = req.limit
    pool_offset = req.offset
    if req.diversify:
        pool_limit = max(req.mmr_pool, req.limit + req.offset)
        pool_offset = 0

    async with pool.acquire() as conn:
        async with conn.transaction(isolation='repeatable_read'):
            # 2. Build Query (Count & Search)
            
            # WHERE Clause Construction
            # tags && $2::text[]  (Contains ANY tag)
            
            count_sql = """
            SELECT COUNT(*)
            FROM documents d
            WHERE d.embedding IS NOT NULL
              AND ($1::text IS NULL OR d.type = $1)
              AND ($2::text[] = '{}' OR d.tags && $2)
              AND ($3::timestamptz IS NULL OR d.created_at >= $3)
              AND ($4::timestamptz IS NULL OR d.created_at <= $4)
            """
            
            count_params = [req.type, req.tags, req.date_from, req.date_to]
            total_row = await conn.fetchrow(count_sql, *count_params)
            total = total_row[0] if total_row else 0
            
            search_sql = """
            SELECT
                d.id, d.type, d.title, d.content, d.source_path, d.chunk_index,
                d.metadata, d.tags, d.created_at,
                d.embedding as embedding,
                1 - (d.embedding <=> $1::vector) as score
            FROM documents d
            WHERE d.embedding IS NOT NULL
              AND ($2::text IS NULL OR d.type = $2)
              AND ($3::text[] = '{}' OR d.tags && $3)
              AND ($4::timestamptz IS NULL OR d.created_at >= $4)
              AND ($5::timestamptz IS NULL OR d.created_at <= $5)
            ORDER BY d.embedding <=> $1::vector ASC
            LIMIT $6 OFFSET $7
            """
            
            search_params = [
                query_vec,
                req.type, req.tags, req.date_from, req.date_to,
                pool_limit, pool_offset
            ]
            
            rows = await conn.fetch(search_sql, *search_params)

    if req.diversify and rows:
        embeddings = [row["embedding"] for row in rows]
        if all(e is not None for e in embeddings):
            scores = [float(row["score"]) if row["score"] else 0.0 for row in rows]
            k_needed = min(len(rows), req.offset + req.limit)
            order = mmr_rerank(scores, embeddings, k=k_needed, lambda_param=req.mmr_lambda)
            rows = [rows[i] for i in order]
        start = req.offset
        end = req.offset + req.limit
        rows = rows[start:end]

    # 3. Transform
    results = []
    for row in rows:
        results.append(DocResult(
            id=str(row["id"]),
            score=float(row["score"]),
            type=row["type"],
            title=row["title"],
            content=row["content"],
            source_path=row["source_path"],
            chunk_index=row["chunk_index"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            tags=row["tags"] or [],
            created_at=row["created_at"]
        ))

    return DocSearchResponse(
        query=req.query,
        total=total,
        results=results
    )
