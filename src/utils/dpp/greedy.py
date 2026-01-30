from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from .kernel import build_kernel, cosine_similarity_matrix, normalize_similarity
from .fallback import mmr_select
from .sampler import k_dpp_sample_exact

# Optional Numba
try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

def dpp_greedy_map_numpy(L: np.ndarray, k: int, eps: float = 1e-12) -> Tuple[List[int], List[float]]:
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    d = np.clip(np.diag(L).copy(), a_min=0.0, a_max=None)  # residuals
    selected: List[int] = []
    increments: List[float] = []
    if n == 0 or k <= 0:
        return selected, increments

    C_cols: List[np.ndarray] = []

    for _ in range(min(k, n)):
        i = int(np.argmax(d))
        gain = d[i]
        if gain <= eps:
            break
        selected.append(i)
        increments.append(float(np.log(max(gain, eps))))

        if not C_cols:
            c_k = L[:, i] / np.sqrt(gain)
        else:
            proj = np.zeros(n, dtype=float)
            for c_j in C_cols:
                proj += c_j * c_j[i]
            c_k = (L[:, i] - proj) / np.sqrt(gain)

        C_cols.append(c_k)
        d = d - np.square(c_k)
        d = np.clip(d, a_min=0.0, a_max=None)

    return selected, increments

# Numba-friendly version (fixed-size storage for C up to k)
def _dpp_greedy_map_numba_impl(L: np.ndarray, k: int, eps: float = 1e-12):
    n = L.shape[0]
    d = np.maximum(np.diag(L).copy(), 0.0)
    selected = -np.ones(k, dtype=np.int64)
    increments = np.zeros(k, dtype=np.float64)
    C = np.zeros((n, k), dtype=np.float64)
    m = 0

    for step in range(min(k, n)):
        i = int(np.argmax(d))
        gain = d[i]
        if gain <= eps:
            break
        selected[m] = i
        increments[m] = np.log(max(gain, eps))

        # compute c_k
        if m == 0:
            C[:, m] = L[:, i] / np.sqrt(gain)
        else:
            proj = np.zeros(n, dtype=np.float64)
            for j in range(m):
                proj += C[:, j] * C[i, j]
            C[:, m] = (L[:, i] - proj) / np.sqrt(gain)

        d = d - C[:, m] * C[:, m]
        d = np.maximum(d, 0.0)
        m += 1

    return selected[:m], increments[:m]

if _NUMBA_AVAILABLE:
    dpp_greedy_map_numba = njit(_dpp_greedy_map_numba_impl)  # type: ignore
else:
    dpp_greedy_map_numba = None  # type: ignore

def dpp_greedy_map(L: np.ndarray, k: int, eps: float = 1e-12, use_numba: bool = False) -> Tuple[List[int], List[float]]:
    if use_numba and _NUMBA_AVAILABLE:
        idx, inc = _dpp_greedy_map_numba_impl(L, k, eps)
        return list(idx.tolist()), list(inc.tolist())
    return dpp_greedy_map_numpy(L, k, eps)

def select_dpp_greedy(
    candidate_ids: List[str],
    qualities: np.ndarray,
    k: int,
    S: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
) -> Dict:
    """
    High-level wrapper with diffusion kernels, personalization, Numba/JAX, and k-DPP.
      params:
        - algorithm: "greedy_map" (default) | "k_dpp"
        - kernel_type: "basic" | "heat" | "random_walk"
        - kernel_params: {heat_t, rw_alpha, rw_steps, use_jax}
        - theta, q_power, epsilon
        - bias_weight, rank_one_gamma
        - use_numba: bool
        - mmr_lambda, eps_greedy
    """
    params = params or {}
    theta = float(params.get("theta", 0.7))
    q_power = float(params.get("q_power", 1.0))
    epsilon = float(params.get("epsilon", 1e-8))
    mmr_lambda = float(params.get("mmr_lambda", 0.5))
    eps_greedy = float(params.get("eps_greedy", 1e-12))
    algorithm = str(params.get("algorithm", "greedy_map"))
    kernel_type = str(params.get("kernel_type", "basic"))
    kernel_params = dict(params.get("kernel_params", {}))
    use_numba = bool(params.get("use_numba", False))
    use_jax = bool(kernel_params.get("use_jax", False))
    bias_weight = float(params.get("bias_weight", 0.0))
    rank_one_gamma = float(params.get("rank_one_gamma", 0.0))

    n = len(candidate_ids)
    assert len(qualities) == n

    if S is None:
        assert embeddings is not None, "Either S or embeddings must be provided"
        S = cosine_similarity_matrix(np.asarray(embeddings, dtype=float))
    else:
        S = np.asarray(S, dtype=float)

    S = normalize_similarity(S, clamp=True, to_unit=True, ensure_diag_one=True)

    # Optional per-candidate personalization (supplied by caller via params["_bias"])
    # API path populates it from Candidate.bias
    bias_arr = None
    if "_bias" in params and params["_bias"] is not None:
        bias_arr = np.asarray(params["_bias"], dtype=float).reshape(-1)
        assert len(bias_arr) == n

    L = build_kernel(
        np.asarray(qualities, dtype=float),
        S,
        theta=theta,
        q_power=q_power,
        epsilon=epsilon,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        bias=bias_arr,
        bias_weight=bias_weight,
        rank_one_gamma=rank_one_gamma,
        use_jax=use_jax,
    )

    error = ""
    if algorithm == "k_dpp":
        try:
            indices = k_dpp_sample_exact(L, k=k, seed=int(params.get("seed", 0)) if "seed" in params else None)
            increments: List[float] = []  # sampling은 로그증분의 개념이 없음
            method = "K_DPP"
        except Exception as e:
            # Fallback to Greedy
            error = f"k-DPP failed: {e}"
            indices, increments = dpp_greedy_map(L, k=k, eps=eps_greedy, use_numba=use_numba)
            method = "DPP_GREEDY"
    else:
        try:
            indices, increments = dpp_greedy_map(L, k=k, eps=eps_greedy, use_numba=use_numba)
            method = "DPP_GREEDY"
        except Exception as e:
            indices, increments = [], []
            error = str(e)

    # Final fallback if selection too short or bad increments
    if len(indices) < min(k, n) or (method == "DPP_GREEDY" and any((not np.isfinite(x)) or x <= 0 for x in increments)):
        mmr_idx = mmr_select(
            qualities=np.asarray(qualities, dtype=float),
            S=S,
            k=min(k, n),
            lambda_param=mmr_lambda,
        )
        indices = mmr_idx
        increments = []
        method = "MMR"

    selected_ids = [candidate_ids[i] for i in indices]

    result = {
        "method": method,
        "selected_indices": indices,
        "selected_ids": selected_ids,
        "increments": increments,
        "params": {
            "theta": theta,
            "q_power": q_power,
            "epsilon": epsilon,
            "mmr_lambda": mmr_lambda,
            "eps_greedy": eps_greedy,
            "algorithm": algorithm,
            "kernel_type": kernel_type,
            "kernel_params": kernel_params,
            "bias_weight": bias_weight,
            "rank_one_gamma": rank_one_gamma,
            "use_numba": use_numba,
        },
        "n_candidates": n,
        "k": k,
        "error": error,
    }
    return result
