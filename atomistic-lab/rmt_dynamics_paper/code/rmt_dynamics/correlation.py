"""Build the atom-wise velocity correlation matrix C.

The paper defines

    C_{ij} = ∫_0^∞ dt  <v_i(t) · v_j(0)>.

At equilibrium with time-reversal symmetry this equals half the two-sided
integral, which is the ω=0 slice of the cross-power spectral density —
manifestly real symmetric and PSD. We estimate it via a **Bartlett**
(triangular) lag window of extent t_max on each side of τ=0. The Bartlett
window's Fourier transform is the non-negative Fejér kernel, so the Parseval
form F^H · diag(W) · F is Hermitian PSD, and the resulting C has all
non-negative eigenvalues — a prerequisite for the von Neumann entropy.

Overall scale is calibrated so a single Cartesian component gives
`diag(C)_i ≈ ∫_0^∞ <v_i(t)·v_i(0)> dt` for trajectories whose velocity
autocorrelation decays well within t_max. Marchenko–Pastur fits with
σ² = trace(C)/N, von Neumann entropy (depends on normalised p_k), and
Δ^λ ratios in transport-coefficient decompositions are all scale-invariant,
so any residual constant offset is harmless downstream.
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_C", "velocity_autocorr_integrals"]


def _bartlett_weight_rfft(
    n_frames: int, pad_len: int, n_lag: int, dt: float
) -> np.ndarray:
    """rfft of the symmetric Bartlett (triangular) window of half-extent n_lag.

    w(|τ|) = dt · (1 − |τ|/n_lag) for |τ| ≤ n_lag, zero otherwise.

    The FT is the Fejér kernel, which is non-negative everywhere. The returned
    array is compensated for one-sided rfft so that a single dot product
    `Σ_k W[k] conj(CC[k])` reproduces the full-spectrum Parseval sum
    `Σ_τ w[τ] cc[τ]`.
    """
    if n_lag < 1:
        raise ValueError("n_lag must be >= 1")
    w = np.zeros(pad_len, dtype=np.float64)
    taus = np.arange(n_lag + 1, dtype=np.float64)
    w_pos = dt * (1.0 - taus / float(n_lag))
    w[: n_lag + 1] = w_pos
    if n_lag >= 1:
        w[pad_len - n_lag : pad_len] = w_pos[n_lag:0:-1]
    W = np.fft.rfft(w).real.astype(np.float64)
    # Fejér kernel is non-negative; clip tiny numerical negatives.
    W = np.maximum(W, 0.0)
    if pad_len % 2 == 0:
        W[1:-1] *= 2.0
    else:
        W[1:] *= 2.0
    return W


def build_C(
    velocities: np.ndarray,
    dt: float,
    t_max: float,
    component_average: bool = True,
    remove_mean: bool = True,
    block: int | None = None,
) -> np.ndarray:
    """Construct the atom-wise velocity correlation matrix.

    Parameters
    ----------
    velocities
        Array of shape (n_frames, N, 3).
    dt
        Time between consecutive sampled frames (physical units).
    t_max
        Upper integration limit for the correlation integral. Rounded down to
        the nearest multiple of `dt`. Must satisfy 0 < t_max <= n_frames * dt.
    component_average
        If True (default), average the three Cartesian-component matrices;
        otherwise return the sum across components.
    remove_mean
        If True (default), subtract the per-particle, per-component time mean
        before correlating. Removes residual DC and matches the convention
        used by the Marchenko–Pastur reference.
    block
        Optional particle-block size used to reduce peak memory for large N.
        When set, the final Gram step is tiled into `block × block` blocks
        instead of materialising the full (N, N) complex intermediate.

    Returns
    -------
    C : np.ndarray, shape (N, N), float64, symmetric.
    """
    v = np.asarray(velocities)
    if v.ndim != 3 or v.shape[-1] != 3:
        raise ValueError(f"velocities must have shape (n_frames, N, 3); got {v.shape}")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not np.isfinite(t_max) or t_max <= 0:
        raise ValueError("t_max must be positive and finite")

    n_frames, n_particles, _ = v.shape
    n_lag = int(np.floor(t_max / dt))
    if n_lag < 1:
        raise ValueError("t_max is smaller than dt; no lag steps included")
    if n_lag > n_frames:
        raise ValueError("t_max exceeds trajectory length")

    pad_len = 2 * n_frames
    W = _bartlett_weight_rfft(n_frames, pad_len, n_lag, dt)
    # Normalisation factor so that diag(C)_i ≈ σ_i^2 · τ_int for smooth
    # processes (matching the paper's ∫_0^∞ convention). The factor is
    # 1/(pad_len · n_frames · 2) — 1/pad_len from rfft Parseval, 1/n_frames
    # from converting the unnormalised cross-correlation Σ_s v(s+τ) v(s)
    # to the "sample mean" form, and 1/2 from converting the symmetric
    # (two-sided) integral to the one-sided integral.
    norm = 1.0 / (float(pad_len) * float(n_frames) * 2.0)

    C = np.zeros((n_particles, n_particles), dtype=np.float64)

    for alpha in range(3):
        V = v[..., alpha].astype(np.float64, copy=False)  # (T, N)
        if remove_mean:
            V = V - V.mean(axis=0, keepdims=True)
        pad = np.zeros((pad_len - n_frames, n_particles), dtype=np.float64)
        Vp = np.concatenate([V, pad], axis=0)
        F = np.fft.rfft(Vp, axis=0)  # (Ω, N) complex
        FW = F * W[:, None]

        if block is None:
            # C^α = norm · Re[F^H diag(W) F]. Fejér kernel => PSD.
            C_alpha = np.real(F.conj().T @ FW) * norm
        else:
            C_alpha = np.zeros((n_particles, n_particles), dtype=np.float64)
            b = int(block)
            for i0 in range(0, n_particles, b):
                i1 = min(i0 + b, n_particles)
                Fi_H = F[:, i0:i1].conj().T
                for j0 in range(0, n_particles, b):
                    j1 = min(j0 + b, n_particles)
                    C_alpha[i0:i1, j0:j1] = np.real(Fi_H @ FW[:, j0:j1]) * norm

        C += C_alpha

    if component_average:
        C /= 3.0

    # Symmetrize to clean up round-off; for real W the result is already PSD
    # up to floating-point noise.
    C = 0.5 * (C + C.T)
    return C.astype(np.float64, copy=False)


def velocity_autocorr_integrals(
    velocities: np.ndarray,
    dt: float,
    t_max: float,
    component_average: bool = True,
    remove_mean: bool = True,
) -> np.ndarray:
    """Integrated single-particle velocity autocorrelation, one entry per particle.

    Returns an array of shape (N,) satisfying `diag(build_C(...)) == this` to
    numerical precision. Useful as a sanity check and as `f_α` building blocks.
    """
    v = np.asarray(velocities)
    if v.ndim != 3 or v.shape[-1] != 3:
        raise ValueError(f"velocities must have shape (n_frames, N, 3); got {v.shape}")

    n_frames, n_particles, _ = v.shape
    n_lag = int(np.floor(t_max / dt))
    if n_lag < 1:
        raise ValueError("t_max is smaller than dt; no lag steps included")

    pad_len = 2 * n_frames
    W = _bartlett_weight_rfft(n_frames, pad_len, n_lag, dt)
    # Normalisation factor so that diag(C)_i ≈ σ_i^2 · τ_int for smooth
    # processes (matching the paper's ∫_0^∞ convention). The factor is
    # 1/(pad_len · n_frames · 2) — 1/pad_len from rfft Parseval, 1/n_frames
    # from converting the unnormalised cross-correlation Σ_s v(s+τ) v(s)
    # to the "sample mean" form, and 1/2 from converting the symmetric
    # (two-sided) integral to the one-sided integral.
    norm = 1.0 / (float(pad_len) * float(n_frames) * 2.0)

    result = np.zeros(n_particles, dtype=np.float64)
    for alpha in range(3):
        V = v[..., alpha].astype(np.float64, copy=False)
        if remove_mean:
            V = V - V.mean(axis=0, keepdims=True)
        pad = np.zeros((pad_len - n_frames, n_particles), dtype=np.float64)
        Vp = np.concatenate([V, pad], axis=0)
        F = np.fft.rfft(Vp, axis=0)
        power = (F * F.conj()).real
        result += (W[:, None] * power).sum(axis=0) * norm

    if component_average:
        result /= 3.0
    return result
