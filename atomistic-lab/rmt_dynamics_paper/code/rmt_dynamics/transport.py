"""Independent transport-coefficient helpers.

`green_kubo_integral` is a deliberately separate code path from `build_C`:
it computes the velocity autocorrelation in the time domain via
`scipy.signal.correlate` and trapezoidally integrates the result. Comparing
its output against `Σ_ij z_i z_j C_ij` (the matrix-based GK route in
Experiment 5) tests the FFT/Bartlett pipeline in `build_C` against a
textbook reference rather than re-deriving an algebraic identity.

`radial_distribution` and `coordination_number` provide the structural
RDF cross-check used in the same experiment.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate

__all__ = [
    "green_kubo_integral",
    "radial_distribution",
    "time_averaged_rdf",
    "coordination_number",
    "contact_graph",
]


def green_kubo_integral(
    signal: np.ndarray,
    dt: float,
    t_max: float,
    remove_mean: bool = True,
) -> float:
    """Component-averaged integrated autocorrelation of a vector-valued signal.

    Returns
    -------
    (1/d) · Σ_α ∫_0^{t_max} <s_α(t) s_α(0)> dt

    using the unbiased estimator (1 / (T - τ)) at each lag and trapezoidal
    integration in time. The component-averaging convention matches
    ``build_C(..., component_average=True)``.

    Implementation deliberately avoids any FFT path so it is independent of
    the Wiener–Khinchin / Bartlett-window machinery in `correlation.py`.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2:
        raise ValueError(f"signal must be 1- or 2-D; got shape {signal.shape}")
    n_frames, d = signal.shape
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_lag = int(np.floor(t_max / dt))
    if n_lag < 1:
        raise ValueError("t_max < dt")
    if n_lag >= n_frames:
        raise ValueError("t_max exceeds trajectory length")

    if remove_mean:
        signal = signal - signal.mean(axis=0, keepdims=True)

    acf = np.zeros(n_lag + 1, dtype=np.float64)
    taus = np.arange(n_lag + 1)
    for k in range(d):
        full = correlate(signal[:, k], signal[:, k], mode="full")
        center = n_frames - 1
        ac = full[center : center + n_lag + 1] / (n_frames - taus)
        acf += ac
    acf /= float(d)
    taus_t = taus * dt
    return float(np.trapezoid(acf, taus_t))


def radial_distribution(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    box: np.ndarray,
    r_max: float,
    n_bins: int = 100,
    same_species: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-frame radial distribution function g_AB(r) under cubic PBC.

    Parameters
    ----------
    positions_a, positions_b
        (N_A, 3) and (N_B, 3) atom positions in Å.
    box
        (3,) cubic box lengths.
    r_max
        Upper bin edge; should be < box.min() / 2 for the minimum-image
        convention to remain valid.
    n_bins
        Number of histogram bins.
    same_species
        If True, treat A and B as the same set (skip self-pairs, count each
        pair once).

    Returns
    -------
    r_centers : (n_bins,) bin centres in Å.
    g : (n_bins,) g_AB(r) values.
    """
    positions_a = np.asarray(positions_a, dtype=np.float64)
    positions_b = np.asarray(positions_b, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)

    if r_max >= 0.5 * box.min():
        raise ValueError("r_max must be < min(box) / 2 for minimum image")

    n_a = positions_a.shape[0]
    n_b = positions_b.shape[0]
    if n_a == 0 or n_b == 0:
        edges = np.linspace(0.0, r_max, n_bins + 1)
        return 0.5 * (edges[:-1] + edges[1:]), np.zeros(n_bins)

    dr = positions_a[:, None, :] - positions_b[None, :, :]
    dr -= box * np.round(dr / box)
    d = np.linalg.norm(dr, axis=-1)
    if same_species:
        # Drop diagonal so a particle isn't counted as its own neighbour.
        np.fill_diagonal(d, np.inf)
        d = d[np.triu_indices(n_a, k=1)]
    else:
        d = d.ravel()

    edges = np.linspace(0.0, r_max, n_bins + 1)
    counts, _ = np.histogram(d, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dr_bin = edges[1] - edges[0]
    shell_vol = 4.0 * np.pi * centres ** 2 * dr_bin
    volume = float(np.prod(box))
    if same_species:
        # n_pairs = N(N-1)/2; ideal density per pair = (n_pairs / V) / (n_pairs).
        ideal = shell_vol * (n_a - 1) / (2.0 * volume)
        norm = ideal * n_a
    else:
        ideal = shell_vol * n_b / volume
        norm = ideal * n_a
    g = np.where(norm > 0, counts / norm, 0.0)
    return centres, g


def time_averaged_rdf(
    pos_a_frames: np.ndarray,
    pos_b_frames: np.ndarray,
    box: np.ndarray,
    r_max: float,
    n_bins: int = 100,
    same_species: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average `radial_distribution` over a sequence of frames."""
    pos_a_frames = np.asarray(pos_a_frames)
    pos_b_frames = np.asarray(pos_b_frames)
    n_frames = pos_a_frames.shape[0]
    if n_frames == 0:
        raise ValueError("need at least one frame")
    centres, g_sum = radial_distribution(
        pos_a_frames[0], pos_b_frames[0], box, r_max, n_bins, same_species,
    )
    for f in range(1, n_frames):
        _, g = radial_distribution(
            pos_a_frames[f], pos_b_frames[f], box, r_max, n_bins, same_species,
        )
        g_sum += g
    return centres, g_sum / float(n_frames)


def coordination_number(
    r: np.ndarray, g: np.ndarray, density_b: float, r_cut: float,
) -> float:
    """N_AB(r_cut) = 4π ρ_B ∫_0^{r_cut} g(r) r² dr (trapezoidal)."""
    mask = r <= r_cut
    if not mask.any():
        return 0.0
    return float(
        4.0 * np.pi * density_b * np.trapezoid(g[mask] * r[mask] ** 2, r[mask])
    )


def contact_graph(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    box: np.ndarray,
    r_cut: float,
) -> np.ndarray:
    """(N_A, N_B) 0/1 adjacency: 1 iff min-image distance < r_cut."""
    positions_a = np.asarray(positions_a, dtype=np.float64)
    positions_b = np.asarray(positions_b, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    dr = positions_a[:, None, :] - positions_b[None, :, :]
    dr -= box * np.round(dr / box)
    d = np.linalg.norm(dr, axis=-1)
    return (d < r_cut).astype(np.int8)
