from __future__ import annotations
import numpy as np
from typing import Optional, Dict

# Optional JAX
try:
    import jax.numpy as jnp  # type: ignore
    from jax import jit  # type: ignore
    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False

def symmetrize(S: np.ndarray) -> np.ndarray:
    return 0.5 * (S + S.T)

def normalize_similarity(S: np.ndarray, clamp: bool = True, to_unit: bool = True, ensure_diag_one: bool = True) -> np.ndarray:
    S = symmetrize(np.asarray(S, dtype=float))
    if clamp:
        S = np.clip(S, -1.0, 1.0)
    if to_unit:
        # [-1,1] -> [0,1]
        S = 0.5 * (S + 1.0)
    if ensure_diag_one:
        np.fill_diagonal(S, 1.0)
    return S

def cosine_similarity_matrix(X: np.ndarray, zero_diag: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    Xn = X / safe
    S = Xn @ Xn.T
    if zero_diag:
        np.fill_diagonal(S, 0.0)
    return S

def _degree_vector(S: np.ndarray) -> np.ndarray:
    # Assuming S >= 0 after normalization to [0,1]
    d = np.sum(np.clip(S, a_min=0.0, a_max=None), axis=1)
    return d

def _normalized_laplacian(S: np.ndarray) -> np.ndarray:
    # L = I - D^{-1/2} S D^{-1/2}
    n = S.shape[0]
    d = _degree_vector(S)
    inv_sqrt = np.zeros_like(d)
    mask = d > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = np.diag(inv_sqrt)
    A = D_inv_sqrt @ S @ D_inv_sqrt
    L = np.eye(n) - symmetrize(A)
    return L

def heat_diffusion_kernel(S: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Heat kernel: H_t = exp(-t * L_norm), L_norm = I - D^{-1/2} S D^{-1/2}
    """
    S = normalize_similarity(S, clamp=True, to_unit=True, ensure_diag_one=True)
    Lnorm = _normalized_laplacian(S)
    # eigh on symmetric PSD
    vals, vecs = np.linalg.eigh(Lnorm)
    exp_vals = np.exp(-t * np.clip(vals, a_min=0.0, a_max=None))
    H = (vecs * exp_vals) @ vecs.T  # vecs @ diag(exp_vals) @ vecs.T
    H = symmetrize(H)
    np.fill_diagonal(H, 1.0)
    H = np.clip(H, 0.0, 1.0)
    return H

def random_walk_kernel(S: np.ndarray, alpha: float = 0.5, steps: int = 10) -> np.ndarray:
    """
    Truncated geometric sum of random-walk transitions:
      P = D^{-1} S, K = sum_{i=0}^{steps} alpha^i P^i
    steps=0 => I
    """
    S = normalize_similarity(S, clamp=True, to_unit=True, ensure_diag_one=True)
    n = S.shape[0]
    d = _degree_vector(S)
    inv = np.zeros_like(d)
    mask = d > 0
    inv[mask] = 1.0 / d[mask]
    D_inv = np.diag(inv)
    P = D_inv @ S  # row-stochastic for rows with d>0

    K = np.eye(n)
    P_power = np.eye(n)
    a = 1.0
    for _ in range(steps):
        P_power = P_power @ P
        a *= alpha
        K = K + a * P_power

    K = symmetrize(K)
    np.fill_diagonal(K, 1.0)
    # Scale to [0,1] if needed
    K = K - np.min(K)
    denom = np.max(K)
    if denom > 0:
        K = K / denom
    return K

def apply_personalization_to_q(q: np.ndarray, bias: Optional[np.ndarray], bias_weight: float = 0.0, mode: str = "multiplicative") -> np.ndarray:
    q = np.asarray(q, dtype=float).flatten()
    if bias is None or bias_weight == 0.0:
        return q
    b = np.asarray(bias, dtype=float).flatten()
    assert len(b) == len(q), "bias length must match q"
    if mode == "multiplicative":
        return q * (1.0 + bias_weight * b)
    else:
        return q + bias_weight * b

def mix_rank_one_personalization(S: np.ndarray, bias: Optional[np.ndarray], gamma: float = 0.0) -> np.ndarray:
    if bias is None or gamma == 0.0:
        return S
    b = np.asarray(bias, dtype=float).flatten()
    if np.allclose(b, 0.0):
        return S
    # Normalize bias to unit L2 to form v v^T
    norm = float(np.linalg.norm(b))
    if norm == 0.0:
        return S
    v = b / norm
    R = np.outer(v, v)
    S2 = (1.0 - gamma) * S + gamma * R
    S2 = symmetrize(S2)
    np.fill_diagonal(S2, 1.0)
    S2 = np.clip(S2, 0.0, 1.0)
    return S2

def build_kernel(
    q: np.ndarray,
    S: np.ndarray,
    theta: float = 0.7,
    q_power: float = 1.0,
    epsilon: float = 1e-8,
    *,
    kernel_type: str = "basic",  # "basic" | "heat" | "random_walk"
    kernel_params: Optional[Dict] = None,
    bias: Optional[np.ndarray] = None,
    bias_weight: float = 0.0,
    rank_one_gamma: float = 0.0,
    use_jax: bool = False,
) -> np.ndarray:
    """
    General kernel builder with diffusion and personalization.
      - kernel_type:
          - "basic": A = theta*S + (1-theta)*I
          - "heat":  A = heat_diffusion_kernel(S, t)
          - "random_walk": A = random_walk_kernel(S, alpha, steps)
      - personalization:
          - q' = q * (1 + bias_weight * bias)
          - S' = (1-gamma) S + gamma * v v^T, v = bias / ||bias||
      - L = D(q'^q_power) @ A @ D(q'^q_power) + epsilon I
    """
    kernel_params = kernel_params or {}
    q = np.asarray(q, dtype=float).flatten()
    S = np.asarray(S, dtype=float)
    n = len(q)
    assert S.shape == (n, n), f"S shape {S.shape} must be ({n},{n})"

    # Base similarity (diffusion)
    if kernel_type == "heat":
        t = float(kernel_params.get("heat_t", 1.0))
        if use_jax and _JAX_AVAILABLE:
            # JAX path for heat kernel
            S1 = normalize_similarity(S, clamp=True, to_unit=True, ensure_diag_one=True)
            Ln = _normalized_laplacian(S1)
            # Move to jax
            Ln_j = jnp.asarray(Ln)
            vals, vecs = jnp.linalg.eigh(Ln_j)
            exp_vals = jnp.exp(-t * jnp.clip(vals, a_min=0.0))
            H = (vecs * exp_vals) @ vecs.T
            A = np.array((H + H.T) * 0.5)
            np.fill_diagonal(A, 1.0)
            A = np.clip(A, 0.0, 1.0)
        else:
            A = heat_diffusion_kernel(S, t=t)
    elif kernel_type == "random_walk":
        alpha = float(kernel_params.get("rw_alpha", 0.5))
        steps = int(kernel_params.get("rw_steps", 10))
        A = random_walk_kernel(S, alpha=alpha, steps=steps)
    else:
        S1 = normalize_similarity(S, clamp=True, to_unit=True, ensure_diag_one=True)
        I = np.eye(n, dtype=float)
        A = theta * S1 + (1.0 - theta) * I

    # Personalization on S
    A = mix_rank_one_personalization(A, bias=bias, gamma=rank_one_gamma)
    # Personalization on q
    q2 = apply_personalization_to_q(q, bias=bias, bias_weight=bias_weight, mode="multiplicative")

    D = np.diag(np.power(np.maximum(q2, 0.0), q_power))
    L = D @ A @ D
    if epsilon > 0.0:
        L = L + epsilon * np.eye(n, dtype=float)
    L = symmetrize(L)
    return L
