"""
ComfyUI Memory RAG v1.1 - Search Endpoint
Optimized: 쿼리 정합성 확보 및 인덱스 활용 극대화
"""

import json
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..database import get_pool
from ..embedder import generate_embedding_sync
from ..ranking import mmr_rerank

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    
    # Filters
    model: Optional[str] = None  # workflows.models 배열 안에 있는지 검사
    status: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    # Pagination
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)

    diversify: bool = False
    mmr_lambda: float = Field(0.6, ge=0.0, le=1.0)
    mmr_pool: int = Field(80, ge=10, le=120)

    include_workflow_json: bool = False


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


@router.post("", response_model=SearchResponse)
async def search_workflows(req: SearchRequest):
    """
    Semantic Search for Workflows
    - HNSW Index 활용 (Cosine Distance)
    - models 필터는 GIN Index 활용
    """
    pool = await get_pool()

    # 1. Query Embedding (CPU Bound -> ThreadPool)
    query_vec = await run_in_threadpool(generate_embedding_sync, req.query)

    # 2. Build Query
    #    HNSW 인덱스를 타게 하려면 ORDER BY 절에 <->, <=>, <#> 연산자가 있어야 함.
    #    vector_cosine_ops를 썼으므로 <=> 사용.
    
    # WHERE 조건절 구성
    # ANY(w.models)는 models가 TEXT[] 타입이므로 정상 작동
    

    pool_limit = req.limit
    pool_offset = req.offset
    if req.diversify:
        pool_limit = max(req.mmr_pool, req.limit + req.offset)
        pool_offset = 0

    async with pool.acquire() as conn:
        async with conn.transaction(isolation='repeatable_read'):
            # 1. Query Embedding (CPU Bound -> ThreadPool)
            # 트랜잭션 내부에서 하되, 이미 위에서 계산했으므로 query_vec 사용
            
            # 2. Total Count (JOIN generations if status filter is active)
            count_sql = """
            SELECT COUNT(DISTINCT w.id)
            FROM workflows w
            LEFT JOIN generations g ON w.id = g.workflow_id
            WHERE w.embedding IS NOT NULL
              AND ($1::text IS NULL OR $1 = ANY(w.models))
              AND ($2::timestamptz IS NULL OR w.created_at >= $2)
              AND ($3::timestamptz IS NULL OR w.created_at <= $3)
              AND ($4::text IS NULL OR g.status = $4)
            """
            
            count_params = [req.model, req.date_from, req.date_to, req.status]
            total_row = await conn.fetchrow(count_sql, *count_params)
            total = total_row[0] if total_row else 0
            
            # 3. Search Results
            search_sql = """
            WITH ranked_gens AS (
                SELECT
                    g.*,
                    ROW_NUMBER() OVER (PARTITION BY g.workflow_id ORDER BY g.created_at DESC) AS rn
                FROM generations g
                WHERE ($4::text IS NULL OR g.status = $4)
            )
            SELECT
                w.id as workflow_id,
                w.workflow_json,
                w.models,
                w.tags,
                w.success_count,
                w.created_at as workflow_created,
                rg.id as generation_id,
                rg.prompt_positive,
                rg.prompt_negative,
                rg.result_paths,
                rg.created_at as generation_created,
                w.embedding as embedding,
                1 - (w.embedding <=> $1::vector) as score
            FROM workflows w
            LEFT JOIN ranked_gens rg ON w.id = rg.workflow_id AND rg.rn = 1
            WHERE w.embedding IS NOT NULL
              AND ($5::text IS NULL OR $5 = ANY(w.models))
              AND ($6::timestamptz IS NULL OR w.created_at >= $6)
              AND ($7::timestamptz IS NULL OR w.created_at <= $7)
              AND ($4::text IS NULL OR rg.id IS NOT NULL) -- status 필터 시 generation 있는 것만
            ORDER BY w.embedding <=> $1::vector ASC
            LIMIT $2 OFFSET $3;
            """
            
            search_params = [
                query_vec, 
                pool_limit,
                pool_offset,
                req.status, 
                req.model, 
                req.date_from, 
                req.date_to
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

    # 3. Transform Results
    results = []
    for row in rows:
        wf_json = None
        if req.include_workflow_json and row["workflow_json"]:
            raw = row["workflow_json"]
            wf_json = json.loads(raw) if isinstance(raw, str) else raw

        results.append(WorkflowResult(
            workflow_id=str(row["workflow_id"]),
            generation_id=str(row["generation_id"]) if row["generation_id"] else None,
            score=float(row["score"]) if row["score"] else 0.0,
            prompt_positive=row["prompt_positive"],
            prompt_negative=row["prompt_negative"],
            models=row["models"] or [],
            tags=row["tags"] or [],
            success_count=row["success_count"] or 0,
            created_at=row["workflow_created"],
            result_paths=row["result_paths"] or [],
            workflow_json=wf_json
        ))

    return SearchResponse(
        query=req.query,
        total=total,
        results=results
    )


@router.get("/quick")
async def quick_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20)
):
    req = SearchRequest(query=q, limit=limit)
    return await search_workflows(req)