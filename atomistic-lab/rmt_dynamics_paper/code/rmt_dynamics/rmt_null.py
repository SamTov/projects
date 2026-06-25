"""Marchenko–Pastur reference and integrated-autocorrelation-time helpers.

For a correlation matrix built from T_eff independent time samples of N
particles with unit per-sample variance σ², the Marchenko–Pastur density is

    ρ(λ) = (1 / (2π λ σ² q)) · √((λ_+ - λ)(λ - λ_-))   for λ ∈ [λ_-, λ_+]

with q = N / T_eff and λ_± = σ² (1 ± √q)². A dirac atom at 0 with mass
(1 - 1/q) is included when q > 1.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

__all__ = [
    "mp_edges",
    "mp_density",
    "mp_cdf",
    "estimate_tau_int",
    "T_eff_from_trajectory",
    "ks_distance",
]


def mp_edges(N: int, T_eff: int | float, sigma2: float = 1.0) -> tuple[float, float]:
    """Return (λ_-, λ_+) for the MP bulk with q = N / T_eff."""
    if N <= 0 or T_eff <= 0:
        raise ValueError("N and T_eff must be positive")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    q = float(N) / float(T_eff)
    sq = np.sqrt(q)
    return float(sigma2 * (1.0 - sq) ** 2), float(sigma2 * (1.0 + sq) ** 2)


def mp_density(
    lam: np.ndarray | float,
    N: int,
    T_eff: int | float,
    sigma2: float = 1.0,
) -> np.ndarray:
    """MP density evaluated at `lam`. Continuous part only; returns 0 outside support."""
    lam_arr = np.asarray(lam, dtype=np.float64)
    lo, hi = mp_edges(N, T_eff, sigma2)
    q = float(N) / float(T_eff)

    out = np.zeros_like(lam_arr)
    mask = (lam_arr > lo) & (lam_arr < hi) & (lam_arr > 0)
    # √((λ_+ - λ)(λ - λ_-)) / (2π λ σ² q)
    val = np.sqrt((hi - lam_arr[mask]) * (lam_arr[mask] - lo))
    out[mask] = val / (2.0 * np.pi * lam_arr[mask] * sigma2 * q)
    return out


def mp_cdf(
    lam: np.ndarray | float,
    N: int,
    T_eff: int | float,
    sigma2: float = 1.0,
) -> np.ndarray:
    """CDF of the MP bulk, including the dirac atom at 0 when q > 1."""
    lam_arr = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    lo, hi = mp_edges(N, T_eff, sigma2)
    q = float(N) / float(T_eff)
    atom_mass = max(0.0, 1.0 - 1.0 / q)  # non-zero only when q > 1

    out = np.zeros_like(lam_arr)
    for idx, x in enumerate(lam_arr):
        if x <= 0:
            out[idx] = 0.0
            continue
        # Atom at 0 (only non-zero for q > 1).
        cum = atom_mass if x > 0 else 0.0
        if x >= hi:
            out[idx] = 1.0
            continue
        if x <= lo:
            out[idx] = cum
            continue
        integral, _ = quad(
            mp_density, lo, x, args=(N, T_eff, sigma2), limit=200,
        )
        out[idx] = cum + integral
    if np.isscalar(lam):
        return out[0]
    return out


def estimate_tau_int(
    velocities: np.ndarray,
    dt: float,
    max_lag: int | None = None,
    min_samples: int = 8,
) -> float:
    """Integrated-autocorrelation time τ_int of per-particle velocities, in frames.

    Computes the per-particle, per-component velocity autocorrelation via FFT,
    normalises each series to its zero-lag value, averages across particles and
    components, and integrates the averaged curve using the Sokal automatic
    window: stop at the first lag τ where τ >= c · τ_int(τ) with c = 5.

    Parameters
    ----------
    velocities : (n_frames, N, 3)
    dt : frame spacing (used for the returned time unit).
    max_lag : upper bound for the search window, in frames. Defaults to n_frames // 4.
    min_samples : minimum lags to include before the automatic-window check kicks in.

    Returns
    -------
    tau_int : float, in the same time units as dt.
    """
    v = np.asarray(velocities, dtype=np.float64)
    if v.ndim != 3 or v.shape[-1] != 3:
        raise ValueError(f"velocities must have shape (n_frames, N, 3); got {v.shape}")
    n_frames = v.shape[0]
    if max_lag is None:
        max_lag = n_frames // 4
    max_lag = int(max(min_samples + 1, min(max_lag, n_frames - 1)))

    pad_len = 2 * n_frames
    # Average per-component, per-particle normalised ACF.
    acf_sum = np.zeros(max_lag + 1, dtype=np.float64)
    count = 0
    for alpha in range(3):
        V = v[..., alpha]
        V = V - V.mean(axis=0, keepdims=True)
        pad = np.zeros((pad_len - n_frames, V.shape[1]), dtype=np.float64)
        Vp = np.concatenate([V, pad], axis=0)
        F = np.fft.rfft(Vp, axis=0)
        power = (F * F.conj()).real
        acf = np.fft.irfft(power, n=pad_len, axis=0)[: max_lag + 1]  # (L+1, N)
        # Divide by (n_frames - τ) for the unbiased estimator.
        taus = np.arange(max_lag + 1)
        acf = acf / (n_frames - taus)[:, None]
        # Normalise each particle's ACF by its value at τ = 0.
        zeros = acf[0:1, :]
        valid = zeros[0] > 0
        if not np.any(valid):
            continue
        acf_n = acf[:, valid] / zeros[:, valid]
        acf_sum += acf_n.sum(axis=1)
        count += int(valid.sum())
    if count == 0:
        raise RuntimeError("could not compute autocorrelation: all series were flat")
    mean_acf = acf_sum / count  # normalised, dimensionless

    # Sokal automatic window.
    c = 5.0
    tau_hat = 0.5  # integrate_0 term for normalised ACF (τ = 0 contributes 1/2)
    win = max_lag
    for tau in range(1, max_lag + 1):
        tau_hat += mean_acf[tau]
        if tau < min_samples:
            continue
        if tau >= c * tau_hat:
            win = tau
            break
    # Re-accumulate up to chosen window (tau_hat already holds this for win == tau).
    integrated = 0.5 + float(mean_acf[1 : win + 1].sum())
    return float(dt * integrated)


def T_eff_from_trajectory(
    velocities: np.ndarray,
    dt: float,
    max_lag: int | None = None,
) -> int:
    """Effective independent-sample count T_eff = round(n_frames * dt / τ_int)."""
    tau_int = estimate_tau_int(velocities, dt, max_lag=max_lag)
    n_frames = velocities.shape[0]
    total_time = n_frames * dt
    return max(1, int(round(total_time / tau_int)))


def ks_distance(
    eigvals: np.ndarray,
    N: int,
    T_eff: int | float,
    sigma2: float = 1.0,
    grid: int = 256,
) -> float:
    """Kolmogorov–Smirnov distance between empirical eigenvalue ECDF and MP CDF.

    Evaluated on a grid spanning `[λ_-, λ_+]` of `grid` points. Returns the
    sup-norm difference (standard KS statistic).
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    lo, hi = mp_edges(N, T_eff, sigma2)
    if hi <= lo:
        return 1.0
    xs = np.linspace(lo, hi, grid)
    # Empirical CDF at each grid point.
    n = eigvals.size
    ev_sorted = np.sort(eigvals)
    ecdf = np.searchsorted(ev_sorted, xs, side="right") / n
    theoretical = mp_cdf(xs, N, T_eff, sigma2)
    return float(np.max(np.abs(ecdf - theoretical)))
