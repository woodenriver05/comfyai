"""
ComfyUI Memory RAG v1 - Recall Endpoint
POST /recall/{workflow_id}/push: Outbox에 저장 (PC 보류 모드)

PC 연결 전: recall_outbox 테이블에 적재
PC 연결 후: 별도 sender 데몬이 outbox를 폴링하여 송신 (추후 구현)
"""

import os
import json
import copy
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel

from ..database import get_pool

router = APIRouter()

# ComfyUI API URL (PC) - 추후 사용
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://192.168.1.100:8188")


# ============================================
# Request/Response Models
# ============================================

class PushRequest(BaseModel):
    """Push 요청 (선택적 오버라이드)"""
    # 프롬프트 오버라이드
    prompt_positive: Optional[str] = None
    prompt_negative: Optional[str] = None

    # 파라미터 오버라이드
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None

    # 대상 PC (기본값 사용)
    target_host: Optional[str] = None


class PushResponse(BaseModel):
    """Push 응답"""
    success: bool
    workflow_id: str
    outbox_id: str  # outbox 레코드 ID
    status: str  # queued, sent (현재는 항상 queued)
    message: str
    workflow_json: Optional[dict] = None  # 적용된 JSON (디버깅용)


class OutboxItem(BaseModel):
    """Outbox 항목"""
    id: str
    workflow_id: str
    status: str
    target_host: str
    overrides: dict
    attempts: int
    created_at: datetime
    last_attempt_at: Optional[datetime] = None


class WorkflowInfo(BaseModel):
    """워크플로우 정보"""
    workflow_id: str
    name: Optional[str]
    models: List[str]
    tags: List[str]
    success_count: int
    created_at: datetime
    last_prompt_positive: Optional[str] = None
    last_prompt_negative: Optional[str] = None
    last_seed: Optional[int] = None


# ============================================
# Helper Functions
# ============================================

def apply_overrides(workflow_json: dict, overrides: PushRequest) -> dict:
    """
    워크플로우 JSON에 오버라이드 적용

    ComfyUI 워크플로우 구조:
    {
        "3": {"inputs": {"seed": 123, ...}, "class_type": "KSampler"},
        "6": {"inputs": {"text": "prompt..."}, "class_type": "CLIPTextEncode"},
        ...
    }
    """
    modified = copy.deepcopy(workflow_json)

    # positive/negative 노드 추적
    positive_nodes = []
    negative_nodes = []

    for node_id, node_data in modified.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})

        # KSampler 노드: seed, steps, cfg 오버라이드
        if class_type in ("KSampler", "KSamplerAdvanced"):
            if overrides.seed is not None:
                inputs["seed"] = overrides.seed
            if overrides.steps is not None:
                inputs["steps"] = overrides.steps
            if overrides.cfg_scale is not None:
                inputs["cfg"] = overrides.cfg_scale

            # positive/negative 연결 추적
            pos_link = inputs.get("positive")
            neg_link = inputs.get("negative")
            if isinstance(pos_link, list) and len(pos_link) >= 1:
                positive_nodes.append(str(pos_link[0]))
            if isinstance(neg_link, list) and len(neg_link) >= 1:
                negative_nodes.append(str(neg_link[0]))

    # CLIP Text Encode 노드: 프롬프트 오버라이드
    for node_id, node_data in modified.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})

        if class_type == "CLIPTextEncode":
            if overrides.prompt_positive and node_id in positive_nodes:
                inputs["text"] = overrides.prompt_positive
            elif overrides.prompt_negative and node_id in negative_nodes:
                inputs["text"] = overrides.prompt_negative

    return modified


# ============================================
# Endpoints
# ============================================

@router.get("/outbox/list", response_model=List[OutboxItem])
async def list_outbox(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    Outbox 대기열 조회

    - status: pending, sent, failed
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch("""
                SELECT id, workflow_id, status, target_host, overrides,
                       attempts, created_at, last_attempt_at
                FROM recall_outbox
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, status, limit)
        else:
            rows = await conn.fetch("""
                SELECT id, workflow_id, status, target_host, overrides,
                       attempts, created_at, last_attempt_at
                FROM recall_outbox
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)

    return [
        OutboxItem(
            id=str(row["id"]),
            workflow_id=str(row["workflow_id"]),
            status=row["status"],
            target_host=row["target_host"],
            overrides=json.loads(row["overrides"]) if row["overrides"] else {},
            attempts=row["attempts"],
            created_at=row["created_at"],
            last_attempt_at=row["last_attempt_at"],
        )
        for row in rows
    ]


@router.get("/{workflow_id}", response_model=WorkflowInfo)
async def get_workflow_info(
    workflow_id: uuid.UUID = Path(..., description="워크플로우 UUID")
):
    """
    워크플로우 정보 조회 (push 전 확인용)
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                w.id, w.name, w.models, w.tags, w.success_count, w.created_at,
                (SELECT prompt_positive FROM generations g
                 WHERE g.workflow_id = w.id
                 ORDER BY g.created_at DESC LIMIT 1) as last_positive,
                (SELECT prompt_negative FROM generations g
                 WHERE g.workflow_id = w.id
                 ORDER BY g.created_at DESC LIMIT 1) as last_negative,
                (SELECT seed FROM generations g
                 WHERE g.workflow_id = w.id
                 ORDER BY g.created_at DESC LIMIT 1) as last_seed
            FROM workflows w
            WHERE w.id = $1
        """, workflow_id)

        if not row:
            raise HTTPException(status_code=404, detail=f"워크플로우 없음: {workflow_id}")

        return WorkflowInfo(
            workflow_id=str(row["id"]),
            name=row["name"],
            models=row["models"] or [],
            tags=row["tags"] or [],
            success_count=row["success_count"] or 0,
            created_at=row["created_at"],
            last_prompt_positive=row["last_positive"],
            last_prompt_negative=row["last_negative"],
            last_seed=row.get("last_seed"),
        )


@router.post("/{workflow_id}/push", response_model=PushResponse)
async def push_workflow(
    workflow_id: uuid.UUID = Path(..., description="워크플로우 UUID"),
    req: Optional[PushRequest] = None,
):
    """
    저장된 워크플로우를 Outbox에 추가 (PC 재실행 대기열)

    - recall_outbox 테이블에 적재
    - PC 연결 후 별도 sender 데몬이 outbox를 폴링하여 송신
    """
    pool = await get_pool()
    req = req or PushRequest()

    async with pool.acquire() as conn:
        # 1. 워크플로우 로드
        row = await conn.fetchrow("""
            SELECT workflow_json FROM workflows WHERE id = $1
        """, workflow_id)

        if not row:
            raise HTTPException(status_code=404, detail=f"워크플로우 없음: {workflow_id}")

        # 2. JSON 파싱
        raw_json = row["workflow_json"]
        workflow_json = json.loads(raw_json) if isinstance(raw_json, str) else raw_json

        # 3. 오버라이드 적용 (None이 아닌 값만 체크)
        has_overrides = any(
            x is not None for x in [
                req.prompt_positive, req.prompt_negative,
                req.seed, req.steps, req.cfg_scale
            ]
        )

        if has_overrides:
            workflow_json = apply_overrides(workflow_json, req)

        # 4. 오버라이드 정보 저장용
        overrides_dict = {
            "prompt_positive": req.prompt_positive,
            "prompt_negative": req.prompt_negative,
            "seed": req.seed,
            "steps": req.steps,
            "cfg_scale": req.cfg_scale,
        }
        # None 제거
        overrides_dict = {k: v for k, v in overrides_dict.items() if v is not None}

        # 5. Outbox에 INSERT
        target_host = req.target_host or COMFYUI_API_URL

        outbox_row = await conn.fetchrow("""
            INSERT INTO recall_outbox (workflow_id, workflow_json, overrides, target_host, status)
            VALUES ($1, $2, $3, $4, 'pending')
            RETURNING id
        """,
            workflow_id,
            json.dumps(workflow_json),
            json.dumps(overrides_dict),
            target_host,
        )

        outbox_id = str(outbox_row["id"])

    return PushResponse(
        success=True,
        workflow_id=str(workflow_id),
        outbox_id=outbox_id,
        status="queued",
        message=f"Outbox에 추가됨 (PC 연결 후 송신 예정). outbox_id: {outbox_id}",
        workflow_json=workflow_json if has_overrides else None,
    )


@router.get("/{workflow_id}/json")
async def get_workflow_json(
    workflow_id: uuid.UUID = Path(..., description="워크플로우 UUID")
):
    """
    워크플로우 원본 JSON 반환 (디버깅/미리보기용)
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT workflow_json FROM workflows WHERE id = $1
        """, workflow_id)

        if not row:
            raise HTTPException(status_code=404, detail=f"워크플로우 없음: {workflow_id}")

        raw = row["workflow_json"]
        return json.loads(raw) if isinstance(raw, str) else raw
