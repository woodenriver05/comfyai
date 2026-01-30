from __future__ import annotations
import numpy as np
from typing import List, Optional
from .kernel import cosine_similarity_matrix, normalize_similarity

def mmr_select(
    qualities: np.ndarray,
    S: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    k: int = 10,
    lambda_param: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) fallback.
    score(i) = lambda * quality(i) - (1-lambda) * max_{j in selected} sim(i,j)
    """
    q = np.asarray(qualities, dtype=float).flatten()
    n = len(q)
    if S is None:
        assert embeddings is not None, "S or embeddings required"
        S = cosine_similarity_matrix(np.asarray(embeddings, dtype=float))
    else:
        S = np.asarray(S, dtype=float)
    S = normalize_similarity(S, clamp=True, to_unit=True, ensure_diag_one=True)

    selected: List[int] = []
    candidates = list(range(n))

    # Start with best quality
    i0 = int(np.argmax(q))
    selected.append(i0)
    candidates.remove(i0)

    while len(selected) < min(k, n) and candidates:
        best_i = None
        best_score = -1e18
        for i in candidates:
            max_sim = 0.0 if not selected else float(np.max(S[i, selected]))
            score = lambda_param * q[i] - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        candidates.remove(best_i)

    return selected
