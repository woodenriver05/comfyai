"""
ComfyUI Memory RAG v1 - Embedding Generator
all-MiniLM-L6-v2 모델 사용 (384차원)

주의: model.encode()는 블로킹 연산이므로
FastAPI에서는 run_in_threadpool()로 감싸서 사용할 것
"""

import os
from typing import List, Optional

from sentence_transformers import SentenceTransformer

# 모델 싱글톤
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """임베딩 모델 반환 (지연 로딩)"""
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"📦 Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        print(f"✅ Model loaded (dim={_model.get_sentence_embedding_dimension()})")
    return _model


def generate_embedding_sync(text: str) -> List[float]:
    """
    텍스트를 임베딩 벡터로 변환 (동기 함수)

    FastAPI async 엔드포인트에서 호출할 때는:
        from fastapi.concurrency import run_in_threadpool
        embedding = await run_in_threadpool(generate_embedding_sync, text)
    """
    if not text or not text.strip():
        # 빈 텍스트면 제로 벡터 반환
        dim = int(os.getenv("EMBEDDING_DIM", "384"))
        return [0.0] * dim

    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_embeddings_sync(texts: List[str]) -> List[List[float]]:
    """여러 텍스트를 배치로 임베딩 (동기 함수)"""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


# 하위 호환성을 위한 별칭 (deprecated, threadpool 사용 권장)
generate_embedding = generate_embedding_sync
generate_embeddings = generate_embeddings_sync
