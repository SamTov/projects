"""Shared fixtures for the rmt_dynamics test suite.

The fixtures here generate synthetic velocity trajectories so the tests
don't depend on any MD engine or disk I/O.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def make_white_noise(rng):
    """Factory: create (n_frames, n_particles, 3) i.i.d. N(0, sigma^2) velocities."""
    def _make(n_frames: int, n_particles: int, sigma: float = 1.0) -> np.ndarray:
        arr = rng.standard_normal((n_frames, n_particles, 3)).astype(np.float64)
        if sigma != 1.0:
            arr *= float(sigma)
        return arr
    return _make


@pytest.fixture
def make_ou(rng):
    """Factory: create stationary Ornstein–Uhlenbeck velocities.

    AR(1) with rho = exp(-1 / tau_frames), per-component stationary variance sigma^2.
    Integrated autocorrelation time in units of frames is exactly tau_frames.
    """
    def _make(n_frames: int, n_particles: int, tau_frames: float, sigma: float = 1.0) -> np.ndarray:
        rho = float(np.exp(-1.0 / float(tau_frames)))
        noise_scale = float(sigma * np.sqrt(1.0 - rho * rho))
        v = np.empty((n_frames, n_particles, 3), dtype=np.float64)
        v[0] = rng.standard_normal((n_particles, 3)) * float(sigma)
        for t in range(1, n_frames):
            v[t] = rho * v[t - 1] + noise_scale * rng.standard_normal((n_particles, 3))
        return v
    return _make
