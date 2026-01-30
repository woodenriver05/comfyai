"""
ComfyUI Memory RAG v1 - FastAPI Application
Mac Mini에서 실행: uvicorn src.api.main:app --host 0.0.0.0 --port 7000 --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import ingest, search, recall, knowledge
from .database import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 DB 연결 관리"""
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="ComfyUI Memory RAG",
    description="AI 협업을 위한 ComfyUI 워크플로우 기억 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (PC에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 프로덕션에서는 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(recall.router, prefix="/recall", tags=["Recall"])
app.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge"])


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "ComfyUI Memory RAG v1",
        "docs": "/docs",
        "endpoints": {
            "ingest": "/ingest/workflow",
            "search": "/search",
            "recall": "/recall/{workflow_id}/push",
        },
    }
