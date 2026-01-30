"""
ComfyUI Memory RAG v1 - Database Connection
pgvector 타입 등록 포함
"""

import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector
from dotenv import load_dotenv

load_dotenv()

# 전역 커넥션 풀
_pool: Optional[asyncpg.Pool] = None


def get_database_url() -> str:
    """데이터베이스 URL 반환"""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://raguser:ragpass@localhost:5432/comfyui_rag"
    )


async def _init_connection(conn: asyncpg.Connection):
    """각 커넥션에 pgvector 타입 등록"""
    await register_vector(conn)


async def init_db():
    """DB 커넥션 풀 초기화 (pgvector 타입 자동 등록)"""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            get_database_url(),
            min_size=2,
            max_size=10,
            command_timeout=60,
            init=_init_connection,  # 각 커넥션에 pgvector 등록
        )
        print(f"✅ Database connected (pgvector enabled): {get_database_url().split('@')[1]}")


async def close_db():
    """DB 커넥션 풀 종료"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        print("🔌 Database disconnected")


async def get_pool() -> asyncpg.Pool:
    """커넥션 풀 반환"""
    global _pool
    if _pool is None:
        await init_db()
    return _pool
