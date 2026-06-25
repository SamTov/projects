"""Spectral observables of the correlation matrix C."""
from __future__ import annotations

import numpy as np

__all__ = [
    "eigenvalues",
    "eigendecomposition",
    "vn_entropy",
    "participation_ratio",
    "trace_normalised",
]


def eigenvalues(C: np.ndarray) -> np.ndarray:
    """Sorted-ascending eigenvalues of the real symmetric matrix C."""
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square; got shape {C.shape}")
    return np.linalg.eigvalsh(C)


def eigendecomposition(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (sorted ascending) and corresponding unit eigenvectors.

    Returns
    -------
    mu : (N,) eigenvalues, sorted ascending.
    Phi : (N, N) eigenvectors stacked as columns so that Phi[:, k] is the
          eigenvector for mu[k].
    """
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square; got shape {C.shape}")
    mu, Phi = np.linalg.eigh(C)
    return mu, Phi


def vn_entropy(
    C: np.ndarray,
    eps: float = 1e-12,
    base: float | None = None,
) -> float:
    """Von Neumann entropy S(C) = -Σ_k p_k log p_k with p_k = μ_k / Σ_l μ_l.

    Parameters
    ----------
    eps
        Lower cutoff on normalised eigenvalues. Values with p_k <= eps are
        dropped from the sum (contribute 0 in the log-zero limit).
    base
        If None (default), returns entropy in nats (natural log). Otherwise
        returns -Σ p log_base(p).
    """
    mu = eigenvalues(C)
    tr = mu.sum()
    if tr <= 0:
        # Null or semi-negative matrix — entropy undefined; surface as NaN.
        return float("nan")
    p = mu / tr
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    s = -float(np.sum(p * np.log(p)))
    if base is not None:
        s /= float(np.log(base))
    return s


def participation_ratio(Phi: np.ndarray) -> np.ndarray:
    """Participation ratio per eigenvector column of Phi.

    PR_k = 1 / Σ_i φ_{ik}^4, computed for each column. Returns (N,).
    Values in [1, N]: 1 = fully localised on one site, N = fully delocalised.
    """
    Phi = np.asarray(Phi)
    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2-D; got shape {Phi.shape}")
    denom = np.sum(Phi ** 4, axis=0)
    # Guard against pathological zero columns.
    denom = np.where(denom > 0, denom, np.inf)
    return 1.0 / denom


def trace_normalised(C: np.ndarray) -> np.ndarray:
    """Return C / trace(C). Raises if trace is non-positive."""
    C = np.asarray(C)
    tr = np.trace(C)
    if tr <= 0:
        raise ValueError(f"trace(C) must be positive; got {tr}")
    return C / tr
