"""Tests for rmt_dynamics.rmt_null."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from rmt_dynamics import (
    build_C,
    eigenvalues,
    estimate_tau_int,
    ks_distance,
    mp_cdf,
    mp_density,
    mp_edges,
)


def test_mp_edges_closed_form():
    N, T, sigma2 = 50, 500, 1.5
    q = N / T
    lo, hi = mp_edges(N, T, sigma2)
    assert np.isclose(lo, sigma2 * (1.0 - np.sqrt(q)) ** 2)
    assert np.isclose(hi, sigma2 * (1.0 + np.sqrt(q)) ** 2)


def test_mp_density_normalises_to_one():
    for N, T in [(10, 500), (50, 2000), (200, 4000)]:
        lo, hi = mp_edges(N, T, sigma2=1.0)
        integral, _ = quad(mp_density, lo, hi, args=(N, T, 1.0), limit=200)
        assert np.isclose(integral, 1.0, atol=5e-3), (N, T, integral)


def test_mp_density_zero_outside_support():
    N, T = 40, 2000
    lo, hi = mp_edges(N, T)
    assert mp_density(lo * 0.5, N, T)[()] == 0.0
    assert mp_density(hi * 1.5, N, T)[()] == 0.0


def test_mp_cdf_monotone_and_bounded():
    N, T = 30, 600
    lo, hi = mp_edges(N, T)
    xs = np.linspace(lo * 0.5, hi * 1.5, 20)
    cdf = mp_cdf(xs, N, T)
    assert np.all(np.diff(cdf) >= -1e-9)
    assert np.isclose(cdf[-1], 1.0, atol=5e-3)
    assert cdf[0] >= -1e-12


def test_mp_null_on_white_noise(make_white_noise):
    """End-to-end MP agreement on i.i.d. Gaussian velocities with L = 1.

    With one lag step (t_max == dt), the construction reduces to a sum of
    three Wishart matrices (one per Cartesian component). The exact MP
    prediction uses T_eff = 3 * n_frames and sigma^2 = trace(C)/N.
    """
    n_frames = 2048
    n_particles = 48
    v = make_white_noise(n_frames=n_frames, n_particles=n_particles, sigma=1.0)
    C = build_C(v, dt=1.0, t_max=1.0)
    mu = eigenvalues(C)
    sigma2 = float(np.trace(C) / n_particles)
    T_eff = 3 * n_frames
    ks = ks_distance(mu, n_particles, T_eff, sigma2)
    assert ks < 0.05, f"KS = {ks:.3f} exceeds 0.05"
    # λ_max should not overshoot the MP edge by more than a few percent.
    _, lam_plus = mp_edges(n_particles, T_eff, sigma2)
    assert mu.max() <= 1.15 * lam_plus


def test_tau_int_recovers_ou_constant(make_ou):
    """AR(1) OU velocities: estimated τ_int ≈ τ_frames · dt within ~10 %."""
    tau_true = 20.0  # frames
    v = make_ou(n_frames=20000, n_particles=32, tau_frames=tau_true, sigma=1.0)
    tau_est = estimate_tau_int(v, dt=1.0)
    assert np.isclose(tau_est, tau_true, rtol=0.15), (
        f"tau_est = {tau_est:.3f}, tau_true = {tau_true:.3f}"
    )


def test_tau_int_rejects_bad_shape():
    with pytest.raises(ValueError):
        estimate_tau_int(np.zeros((10, 4)), dt=1.0)
