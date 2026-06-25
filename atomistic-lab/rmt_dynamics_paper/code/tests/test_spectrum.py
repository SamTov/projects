"""Tests for rmt_dynamics.spectrum."""
from __future__ import annotations

import numpy as np
import pytest

from rmt_dynamics import (
    eigendecomposition,
    eigenvalues,
    participation_ratio,
    trace_normalised,
    vn_entropy,
)


def test_eigenvalues_sorted_ascending():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((16, 16))
    C = A @ A.T  # symmetric PSD
    mu = eigenvalues(C)
    assert mu.shape == (16,)
    assert np.all(np.diff(mu) >= -1e-12)


def test_eigendecomposition_reconstructs_C():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 10))
    C = A @ A.T
    mu, Phi = eigendecomposition(C)
    reconstruction = Phi @ np.diag(mu) @ Phi.T
    assert np.allclose(C, reconstruction, atol=1e-10)


def test_vn_entropy_uniform_gives_log_N():
    N = 13
    C = np.eye(N) * 2.7  # scale shouldn't matter — entropy depends on p_k only
    S = vn_entropy(C)
    assert np.isclose(S, np.log(N), atol=1e-12)


def test_vn_entropy_rank_one_is_zero():
    v = np.arange(1, 9, dtype=np.float64)
    C = np.outer(v, v)  # rank-1 PSD
    S = vn_entropy(C)
    assert np.isclose(S, 0.0, atol=1e-10)


def test_vn_entropy_base_conversion():
    N = 8
    C = np.eye(N)
    S_nats = vn_entropy(C)
    S_bits = vn_entropy(C, base=2.0)
    assert np.isclose(S_bits, S_nats / np.log(2.0))


def test_vn_entropy_non_positive_trace_is_nan():
    # Zero matrix has trace 0 and is defined as NaN.
    C = np.zeros((4, 4))
    assert np.isnan(vn_entropy(C))


def test_participation_ratio_bounds():
    # Delocalised: uniform unit vector → PR == N.
    N = 17
    Phi = np.eye(N)  # each column has PR = 1 (fully localised)
    pr_localised = participation_ratio(Phi)
    assert np.allclose(pr_localised, 1.0)

    # Uniform vector column: PR = N.
    delocalised = np.ones((N, 1)) / np.sqrt(N)
    pr_delocalised = participation_ratio(delocalised)
    assert np.isclose(pr_delocalised[0], N)

    # General symmetric random case: 1 ≤ PR ≤ N.
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    pr = participation_ratio(Q)
    assert np.all(pr >= 1.0 - 1e-10)
    assert np.all(pr <= 12.0 + 1e-10)


def test_trace_normalised_sums_to_one():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6))
    C = A @ A.T + np.eye(6)
    Cn = trace_normalised(C)
    assert np.isclose(np.trace(Cn), 1.0, atol=1e-12)
    with pytest.raises(ValueError):
        trace_normalised(np.zeros((5, 5)))
