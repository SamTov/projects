"""Tests for rmt_dynamics.transport (GK, RDF, coordination)."""
from __future__ import annotations

import numpy as np
import pytest

from rmt_dynamics import (
    build_C,
    contact_graph,
    coordination_number,
    green_kubo_integral,
    radial_distribution,
    time_averaged_rdf,
)


def _make_ar1(rng, n_frames, rho, d=3, sigma=1.0):
    noise_scale = float(sigma * np.sqrt(1 - rho * rho))
    sig = np.zeros((n_frames, d))
    sig[0] = rng.standard_normal(d) * sigma
    for t in range(1, n_frames):
        sig[t] = rho * sig[t - 1] + noise_scale * rng.standard_normal(d)
    return sig


def test_gk_recovers_OU_theoretical_integral():
    """Average-over-seeds GK on AR(1) ≈ σ² · τ_int. Single-seed scatter is
    O(20 %) even at 32k frames so we average 5 independent realisations."""
    rho = 0.9
    tau_int = -1.0 / float(np.log(rho))
    dt = 1.0
    t_max = 30.0 * tau_int
    vals = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sig = _make_ar1(rng, n_frames=32768, rho=rho)
        vals.append(green_kubo_integral(sig, dt=dt, t_max=t_max))
    mean = float(np.mean(vals))
    assert np.isclose(mean, tau_int, rtol=0.15), (
        f"mean GK = {mean}, expected ~ {tau_int}, samples = {vals}"
    )


def test_gk_and_build_C_agree_in_long_window_limit():
    """Textbook GK and FFT/Bartlett build_C estimate the same physical
    integral. The Bartlett window's finite-L bias scales as ~τ_int/L; even at
    L ≫ τ_int, single-seed AR(1) scatter is the dominant residual. Average
    across 5 seeds and demand 20 % agreement — this is the genuinely
    independent cross-check on build_C that the matrix-route Δ^σ relies on.
    """
    rho = 0.9
    tau_int = -1.0 / float(np.log(rho))
    dt = 1.0
    t_max = 30.0 * tau_int
    gks = []
    fcs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sig = _make_ar1(rng, n_frames=32768, rho=rho)
        gks.append(green_kubo_integral(sig, dt=dt, t_max=t_max))
        fcs.append(float(build_C(sig[:, None, :], dt=dt, t_max=t_max)[0, 0]))
    gk_mean = float(np.mean(gks))
    fc_mean = float(np.mean(fcs))
    assert np.isclose(gk_mean, fc_mean, rtol=0.2), (
        f"GK mean = {gk_mean}, build_C mean = {fc_mean}"
    )


def test_gk_white_noise_trapezoid_edge():
    """Trapezoidal integration of an autocorrelation that is `σ²` at τ=0
    and ~0 elsewhere returns `dt · σ² / 2` — the τ=0 sample sits at the
    integration boundary so it contributes only half of dt·σ². Documenting
    this so callers don't confuse the trapezoidal GK with the discrete-sum
    "diag of C" convention used elsewhere."""
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((8192, 3))
    dt = 1.0
    val = green_kubo_integral(sig, dt=dt, t_max=10.0)
    assert np.isclose(val, 0.5 * dt, atol=0.05), val


def test_radial_distribution_uniform_random_is_flat_at_one():
    """For an ideal gas (uniform random positions), g(r) ≈ 1 outside r=0."""
    rng = np.random.default_rng(2)
    box = np.array([20.0, 20.0, 20.0])
    n = 4000
    pos = rng.random((n, 3)) * box
    r, g = radial_distribution(pos, pos, box, r_max=9.0, n_bins=40,
                                same_species=True)
    # Skip the first few bins (small-r noise).
    g_bulk = g[5:]
    assert np.isclose(g_bulk.mean(), 1.0, atol=0.1), g_bulk.mean()


def test_radial_distribution_rejects_too_large_rmax():
    box = np.array([10.0, 10.0, 10.0])
    pos = np.zeros((1, 3))
    with pytest.raises(ValueError):
        radial_distribution(pos, pos, box, r_max=6.0, n_bins=20)


def test_coordination_number_integrates_first_shell():
    """Synthetic: a delta-like g(r) at r0=2 with unit area gives N = 4π ρ r0^2."""
    r = np.linspace(0.01, 5.0, 1000)
    g = np.zeros_like(r)
    # Spike centred at r0 = 2.0, normalised so ∫ g r^2 dr ≈ 1.
    sigma = 0.05
    g = np.exp(-((r - 2.0) ** 2) / (2 * sigma ** 2))
    norm = np.trapezoid(g * r ** 2, r)
    g /= norm
    density = 0.1
    expected = 4.0 * np.pi * density * 1.0  # because we normalised ∫ g r² dr = 1
    n_coord = coordination_number(r, g, density_b=density, r_cut=3.0)
    assert np.isclose(n_coord, expected, rtol=0.05)


def test_contact_graph_recovers_nearest_neighbours():
    """A simple chain of 3 atoms along x: each adjacent pair within r_cut."""
    box = np.array([100.0, 100.0, 100.0])
    a_pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    b_pos = np.array([[1.5, 0.0, 0.0], [4.5, 0.0, 0.0]])  # midpoints
    adj = contact_graph(a_pos, b_pos, box, r_cut=2.0)
    # a[0] within 2.0 of b[0] only; a[1] within 2.0 of both; a[2] of b[1] only.
    assert adj[0, 0] == 1
    assert adj[0, 1] == 0
    assert adj[1, 0] == 1
    assert adj[1, 1] == 1
    assert adj[2, 1] == 1


def test_time_averaged_rdf_matches_single_frame_when_static():
    rng = np.random.default_rng(3)
    box = np.array([12.0, 12.0, 12.0])
    n = 500
    pos = rng.random((n, 3)) * box
    r1, g1 = radial_distribution(pos, pos, box, r_max=5.0, n_bins=20, same_species=True)
    frames = np.broadcast_to(pos, (4, *pos.shape)).copy()
    r2, g2 = time_averaged_rdf(frames, frames, box, r_max=5.0, n_bins=20, same_species=True)
    assert np.allclose(r1, r2)
    assert np.allclose(g1, g2)
