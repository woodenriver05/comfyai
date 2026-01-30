"""
ComfyUI Memory RAG v1.1 - Ranking Utilities
Lightweight MMR (Maximal Marginal Relevance) for diversity.
"""

import numpy as np
from typing import List, Dict, Any

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def mmr_select(
    candidates: List[Dict[str, Any]],
    k: int = 5,
    lambda_param: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) for diversity ranking.
    
    Args:
        candidates: List of result dicts. Each must have 'score' and 'embedding'.
        k: Number of items to select.
        lambda_param: 0.0 to 1.0. Higher means more relevance, lower means more diversity.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    selected = []
    remaining = list(candidates)

    # 1. Start with the most relevant candidate (already sorted by DB)
    first = remaining.pop(0)
    selected.append(first)

    while len(selected) < k and remaining:
        best_mmr_score = -float('inf')
        best_idx = -1

        for i, cand in enumerate(remaining):
            # Relevance to query (use the score provided by DB)
            relevance = cand['score']

            # Max similarity with already selected items (Redundancy)
            # cand['embedding'] is expected to be a list or np.ndarray
            cand_vec = np.array(cand['embedding'])
            max_similarity = max([
                cosine_similarity(cand_vec, np.array(s['embedding']))
                for s in selected
            ])

            # MMR Equation: λ * relevance - (1 - λ) * max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        if best_idx != -1:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected

def mmr_rerank(
    relevance_scores: List[float],
    embeddings: List[List[float]],
    k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    """
    MMR rerank returning selected indices.
    Uses cosine similarity between embeddings and relevance scores from DB.
    """
    n = len(relevance_scores)
    if n == 0 or k <= 0:
        return []

    k = min(k, n)
    lambda_param = max(0.0, min(1.0, float(lambda_param)))

    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2 or X.shape[0] != n:
        return list(range(k))

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    Xn = X / norms
    sim = Xn @ Xn.T

    scores = np.asarray(relevance_scores, dtype=float)
    selected: List[int] = []
    candidates = list(range(n))

    first = int(np.argmax(scores))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e18
        for i in candidates:
            max_sim = float(np.max(sim[i, selected])) if selected else 0.0
            score = lambda_param * float(scores[i]) - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        candidates.remove(best_i)

    return selected
