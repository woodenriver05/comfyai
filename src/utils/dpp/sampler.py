from __future__ import annotations
import numpy as np
from typing import List, Optional

def _eigendecompose_psd(L: np.ndarray):
    L = 0.5 * (L + L.T)
    vals, vecs = np.linalg.eigh(L)
    # clip tiny negatives
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return vals, vecs

def _sample_k_eigen_indices(lams: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """
    Sample a size-k subset of eigenvalues indices according to elementary
    symmetric polynomials as in k-DPP.
    """
    n = len(lams)
    # E[i, j] = e_j(lam_1..lam_i)
    E = np.zeros((n + 1, k + 1), dtype=float)
    E[:, 0] = 1.0
    for i in range(1, n + 1):
        lam = lams[i - 1]
        for j in range(1, min(i, k) + 1):
            E[i, j] = E[i - 1, j] + lam * E[i - 1, j - 1]

    J: List[int] = []
    j = k
    for i in range(n, 0, -1):
        if j == 0:
            break
        lam = lams[i - 1]
        if E[i, j] == 0:
            continue
        prob = lam * E[i - 1, j - 1] / E[i, j]
        if rng.random() < prob:
            J.append(i - 1)
            j -= 1
    J.reverse()
    return J

def _orthonormalize_columns(M: np.ndarray) -> np.ndarray:
    # Orthonormalize columns via QR
    Q, _ = np.linalg.qr(M)
    return Q

def k_dpp_sample_exact(L: np.ndarray, k: int, seed: Optional[int] = None) -> List[int]:
    """
    Exact k-DPP sampler for L-ensemble.
      1) Eigendecompose L = U diag(λ) U^T
      2) Sample subset J of eigenvectors (|J|=k) using DP on elementary symmetric polynomials
      3) Iteratively sample items using row norms, orthonormalizing V
    """
    n = L.shape[0]
    if k <= 0:
        return []
    if k > n:
        k = n

    lams, U = _eigendecompose_psd(L)
    # Avoid degenerate case
    if np.count_nonzero(lams) == 0:
        # fallback: uniform k without replacement
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=k, replace=False).tolist()

    rng = np.random.default_rng(seed)
    J = _sample_k_eigen_indices(lams, k, rng)
    if len(J) < k:
        # If DP degenerates due to numeric issues, pick top-k by lambda
        J = np.argsort(lams)[-k:].tolist()

    V = U[:, J]  # n x k
    # Iterative sampling
    Y: List[int] = []
    while V.shape[1] > 0:
        # probabilities proportional to row norms squared
        probs = np.sum(V * V, axis=1)
        s = float(probs.sum())
        if s <= 0:
            # choose remaining uniformly
            remaining = list(set(range(n)) - set(Y))
            pick = rng.choice(remaining)
        else:
            probs = probs / s
            pick = int(rng.choice(n, p=probs))
        Y.append(pick)

        # Select an orthonormal vector with non-zero component at 'pick'
        # Compute the component vector
        v = V[pick, :]
        if np.allclose(v, 0.0):
            # Re-orthonormalize to avoid degeneracy
            V = _orthonormalize_columns(V)
            v = V[pick, :]
            if np.allclose(v, 0.0):
                break

        # Normalize 'v' to form direction u
        u = (V @ v) / np.sqrt(np.dot(v, v))
        # Project V to subspace orthogonal to e_pick direction
        V = V - np.outer(u, V[pick, :])
        # Re-orthonormalize
        if V.shape[1] > 1:
            V = _orthonormalize_columns(V)
        else:
            # Single vector: normalize
            norm = np.linalg.norm(V, axis=0, keepdims=True)
            norm[norm == 0] = 1.0
            V = V / norm

        # Remove vectors that become numerically zero
        keep = [j for j in range(V.shape[1]) if not np.allclose(V[:, j], 0.0)]
        if len(keep) < V.shape[1]:
            V = V[:, keep]

    # Deduplicate in rare numeric failure cases
    Y = list(dict.fromkeys(Y))
    if len(Y) > k:
        Y = Y[:k]
    elif len(Y) < k:
        # Complete with highest diagonal entries
        diag = np.diag(L)
        candidates = [i for i in np.argsort(-diag).tolist() if i not in Y]
        Y.extend(candidates[: (k - len(Y))])

    return Y
