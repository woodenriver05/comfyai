"""
ComfyUI Memory RAG v1.1 - Ingest Endpoint
Optimized: 임베딩 계산을 DB 트랜잭션 밖으로 분리하여 동시성 향상
"""

import hashlib
import json
from typing import List, Optional

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..database import get_pool
from ..embedder import generate_embedding_sync

router = APIRouter()


class IngestParams(BaseModel):
    seed: Optional[int] = None
    cfg_scale: Optional[float] = Field(None, alias="cfg")
    steps: Optional[int] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None


class IngestOutput(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    frames: Optional[int] = None
    duration_sec: Optional[float] = None
    paths: List[str] = []
    thumbnail: Optional[str] = None


class IngestRequest(BaseModel):
    client_run_id: str = Field(..., description="Unique Run ID from Client")
    workflow_json: dict
    prompt_positive: Optional[str] = None
    prompt_negative: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    params: Optional[IngestParams] = None
    output: Optional[IngestOutput] = None
    status: str = "success"
    error_log: Optional[str] = None
    render_time_sec: Optional[float] = None
    workflow_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    workflow_id: str
    generation_id: str
    workflow_hash: str
    status: str
    message: str


def compute_workflow_hash(workflow_json: dict) -> str:
    """SHA-256 (64 chars) - Low collision risk"""
    canonical = json.dumps(workflow_json, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _safe_str(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _iter_nodes_api(workflow_json: dict):
    if not isinstance(workflow_json, dict):
        return
    for _, node in workflow_json.items():
        if isinstance(node, dict):
            yield node


def _iter_nodes_ui(workflow_json: dict):
    nodes = workflow_json.get("nodes") if isinstance(workflow_json, dict) else None
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                yield node


def _extract_node_types(workflow_json: dict) -> set:
    node_types = set()
    for node in _iter_nodes_api(workflow_json):
        ct = node.get("class_type") or node.get("type")
        if ct:
            node_types.add(ct)
    for node in _iter_nodes_ui(workflow_json):
        ct = node.get("class_type") or node.get("type")
        if ct:
            node_types.add(ct)
    return node_types


def _extract_prompts(workflow_json: dict) -> list:
    prompts = []
    for node in _iter_nodes_api(workflow_json):
        ct = (node.get("class_type") or "").lower()
        if "cliptextencode" in ct or "prompt" in ct:
            inputs = node.get("inputs") or {}
            text = inputs.get("text")
            if isinstance(text, str) and text.strip():
                prompts.append(text.strip())
    for node in _iter_nodes_ui(workflow_json):
        ct = (node.get("type") or "").lower()
        if "cliptextencode" in ct or "prompt" in ct:
            values = node.get("widgets_values") or []
            if values and isinstance(values[0], str) and values[0].strip():
                prompts.append(values[0].strip())
    return prompts


def _extract_models(workflow_json: dict) -> list:
    models = []
    name_keys = {"ckpt_name", "checkpoint", "model_name", "vae_name", "clip_name", "unet_name"}
    for node in _iter_nodes_api(workflow_json):
        inputs = node.get("inputs") or {}
        for k, v in inputs.items():
            if k in name_keys and isinstance(v, str) and v.strip():
                models.append(v.strip())
    for node in _iter_nodes_ui(workflow_json):
        ct = (node.get("type") or "").lower()
        values = node.get("widgets_values") or []
        if values and isinstance(values[0], str) and values[0].strip():
            if "loader" in ct or "checkpoint" in ct or "unet" in ct or "vae" in ct:
                models.append(values[0].strip())
    return models


def _extract_video_meta(workflow_json: dict) -> dict:
    meta = {"width": None, "height": None, "fps": None, "frames": None}
    keys = {"width", "height", "fps", "frames"}
    for node in _iter_nodes_api(workflow_json):
        inputs = node.get("inputs") or {}
        for k in keys:
            if k in inputs and meta[k] is None:
                try:
                    meta[k] = int(inputs[k])
                except Exception:
                    pass
    return meta


def _infer_model_family(models: list, node_types: set) -> list:
    families = set()
    hay = " ".join([_safe_str(m).lower() for m in models] + [t.lower() for t in node_types])
    if "wan" in hay:
        families.add("wan")
    if "hunyuan" in hay:
        families.add("hunyuan")
    if "ltx" in hay:
        families.add("ltx")
    if "cogvideo" in hay or "cog" in hay:
        families.add("cogvideo")
    if "sdxl" in hay:
        families.add("sdxl")
    return sorted(families)


def extract_full_text(
    workflow_json: dict,
    p_pos: str,
    p_neg: str,
    models: List[str],
    tags: Optional[List[str]] = None,
    workflow_name: Optional[str] = None,
    output: Optional[IngestOutput] = None,
    params: Optional[IngestParams] = None,
) -> str:
    """검색용 텍스트 추출 (프롬프트/모델/해상도/프레임 포함)"""
    parts = []
    node_types = _extract_node_types(workflow_json)
    prompt_texts = _extract_prompts(workflow_json)
    model_list = list({m for m in (models or []) + _extract_models(workflow_json) if m})

    video_meta = _extract_video_meta(workflow_json)
    if output:
        if output.width is not None:
            video_meta["width"] = output.width
        if output.height is not None:
            video_meta["height"] = output.height
        if output.fps is not None:
            video_meta["fps"] = output.fps
        if output.frames is not None:
            video_meta["frames"] = output.frames
        if video_meta["frames"] is None and output.fps and output.duration_sec:
            try:
                video_meta["frames"] = int(round(output.fps * output.duration_sec))
            except Exception:
                pass

    if node_types:
        parts.append("nodes: " + " ".join(sorted(node_types)))
    if workflow_name:
        parts.append("name: " + workflow_name)
    if tags:
        parts.append("tags: " + " ".join([_safe_str(t) for t in tags if t]))

    if p_pos:
        parts.append(p_pos)
    if p_neg:
        parts.append("negative: " + p_neg)
    if prompt_texts:
        parts.append("prompts: " + " ".join(prompt_texts[:3]))

    if model_list:
        parts.append("models: " + " ".join(model_list))

    if video_meta.get("width") and video_meta.get("height"):
        parts.append("resolution: %sx%s" % (video_meta["width"], video_meta["height"]))
    if video_meta.get("fps"):
        parts.append("fps: %s" % video_meta["fps"])
    if video_meta.get("frames"):
        parts.append("frames: %s" % video_meta["frames"])

    if output and output.duration_sec:
        parts.append("duration: %ss" % output.duration_sec)

    if params:
        if params.steps is not None:
            parts.append("steps: %s" % params.steps)
        if params.sampler:
            parts.append("sampler: " + params.sampler)
        if params.scheduler:
            parts.append("scheduler: " + params.scheduler)

    families = _infer_model_family(model_list, node_types)
    if families:
        parts.append("model_family: " + " ".join(families))

    return " | ".join([p for p in parts if p]) if parts else "empty workflow"


@router.post("/workflow", response_model=IngestResponse)
async def ingest_workflow(req: IngestRequest):
    """
    워크플로우 수집 (Non-blocking Embeddings)
    1. CPU Bound 작업(해시, 텍스트, 임베딩)을 먼저 처리
    2. 그 후 짧게 DB 트랜잭션 수행
    """
    
    # --- CPU Bound Zone (No DB Lock) ---
    
    # 1. 해시 계산
    workflow_hash = compute_workflow_hash(req.workflow_json)

    # 2. 텍스트 추출
    full_text = extract_full_text(
        req.workflow_json,
        req.prompt_positive or "",
        req.prompt_negative or "",
        req.models,
        tags=req.tags,
        workflow_name=req.workflow_name,
        output=req.output,
        params=req.params,
    )

    # 3. 임베딩 생성 (Heavy Task -> ThreadPool)
    #    트랜잭션 진입 전에 미리 계산 완료!
    wf_embedding = await run_in_threadpool(generate_embedding_sync, full_text)
    prompt_embedding = await run_in_threadpool(
        generate_embedding_sync, 
        req.prompt_positive or ""
    )

    # --- I/O Bound Zone (DB Transaction) ---
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 4. Workflows UPSERT
            #    updated_at은 트리거가 자동 처리하거나, 명시적으로 NOW() 지정 가능
            workflow_row = await conn.fetchrow(
                """
                INSERT INTO workflows (workflow_hash, workflow_json, full_text, embedding, models, name, tags)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
                ON CONFLICT (workflow_hash)
                DO UPDATE SET
                    models = EXCLUDED.models,
                    tags = EXCLUDED.tags
                    -- updated_at은 Trigger가 처리함
                RETURNING id, (xmax = 0) AS is_new
                """,
                workflow_hash,
                json.dumps(req.workflow_json),
                full_text,
                wf_embedding,
                req.models,
                req.workflow_name,
                req.tags,
            )
            
            wf_id = str(workflow_row["id"])
            is_new_wf = workflow_row["is_new"]

            # 5. Generations UPSERT
            params = req.params or IngestParams()
            output = req.output or IngestOutput()

            gen_row = await conn.fetchrow(
                """
                INSERT INTO generations (
                    workflow_id, client_run_id,
                    prompt_positive, prompt_negative,
                    seed, cfg_scale, steps, sampler, scheduler,
                    width, height, fps, duration_sec,
                    result_paths, thumbnail_path,
                    status, error_log, render_time_sec,
                    embedding
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19::vector)
                ON CONFLICT (client_run_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    error_log = EXCLUDED.error_log,
                    result_paths = EXCLUDED.result_paths
                RETURNING id, (xmax = 0) AS is_new
                """,
                wf_id,
                req.client_run_id,
                req.prompt_positive,
                req.prompt_negative,
                params.seed,
                params.cfg_scale,
                params.steps,
                params.sampler,
                params.scheduler,
                output.width,
                output.height,
                output.fps,
                output.duration_sec,
                output.paths,
                output.thumbnail,
                req.status,
                req.error_log,
                req.render_time_sec,
                prompt_embedding
            )
            
            gen_id = str(gen_row["id"])
            is_new_gen = gen_row["is_new"]

    if is_new_wf and is_new_gen:
        status = "created"
        msg = "New workflow and run saved."
    elif is_new_gen:
        status = "created"
        msg = "New run added to existing workflow."
    else:
        status = "updated"
        msg = "Existing run updated."

    return IngestResponse(
        workflow_id=wf_id,
        generation_id=gen_id,
        workflow_hash=workflow_hash,
        status=status,
        message=msg
    )
