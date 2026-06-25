"""Tests for rmt_dynamics.correlation."""
from __future__ import annotations

import numpy as np

from rmt_dynamics import build_C, velocity_autocorr_integrals


def test_C_symmetric(make_white_noise):
    v = make_white_noise(n_frames=1024, n_particles=32)
    C = build_C(v, dt=1.0, t_max=5.0)
    assert C.shape == (32, 32)
    assert np.allclose(C, C.T, atol=1e-10)


def test_C_trace_matches_vacf(make_white_noise):
    v = make_white_noise(n_frames=1024, n_particles=32)
    C = build_C(v, dt=1.0, t_max=5.0)
    diag = np.diag(C)
    integrals = velocity_autocorr_integrals(v, dt=1.0, t_max=5.0)
    assert np.allclose(diag, integrals, rtol=1e-8, atol=1e-12)


def test_C_is_float64(make_white_noise):
    v = make_white_noise(n_frames=256, n_particles=8)
    C = build_C(v, dt=1.0, t_max=2.0)
    assert C.dtype == np.float64


def test_C_block_matches_unblocked(make_white_noise):
    v = make_white_noise(n_frames=256, n_particles=16)
    C_full = build_C(v, dt=1.0, t_max=3.0, block=None)
    C_blocked = build_C(v, dt=1.0, t_max=3.0, block=4)
    assert np.allclose(C_full, C_blocked, atol=1e-10)


def test_C_mean_white_noise_is_identity_scaled(make_white_noise):
    # L = 1 (t_max = dt), Bartlett window, component-averaged.
    # Convention matches the continuous ∫_0^∞ integral: for a δ-like
    # velocity autocorrelation, ∫_0^∞ δ(τ) dτ = 1/2, so diag ~ dt*σ^2 / 2.
    n_frames = 4096
    n_particles = 48
    sigma = 1.0
    dt = 1.0
    v = make_white_noise(n_frames=n_frames, n_particles=n_particles, sigma=sigma)
    C = build_C(v, dt=dt, t_max=dt)

    diag = np.diag(C)
    expected = 0.5 * dt * sigma ** 2
    assert np.allclose(diag.mean(), expected, rtol=0.05)
    # Off-diagonal small compared to diagonal (sampling noise ~ sqrt(N/T)).
    off = C - np.diag(diag)
    assert np.abs(off).max() < 0.3 * diag.mean()


def test_C_rejects_wrong_shape():
    import pytest

    with pytest.raises(ValueError):
        build_C(np.zeros((10, 5)), dt=1.0, t_max=1.0)
    with pytest.raises(ValueError):
        build_C(np.zeros((10, 5, 2)), dt=1.0, t_max=1.0)
    with pytest.raises(ValueError):
        build_C(np.zeros((10, 5, 3)), dt=1.0, t_max=0.0)
    with pytest.raises(ValueError):
        build_C(np.zeros((10, 5, 3)), dt=1.0, t_max=100.0)
