"""Tests for rmt_dynamics.peaks."""
from __future__ import annotations

import numpy as np

from rmt_dynamics import find_peak_in_window, fwhm_in_window


def test_find_peak_synthetic_lorentzian():
    x = np.linspace(-5, 5, 401)
    # Lorentzian peaked at x = 1 with FWHM = 1.0 (half-width = 0.5).
    y = 1.0 / (1.0 + ((x - 1.0) / 0.5) ** 2)
    peak_x, peak_h = find_peak_in_window(x, y, (-5.0, 5.0))
    assert np.isclose(peak_x, 1.0, atol=0.05)
    assert np.isclose(peak_h, 1.0, atol=0.01)


def test_fwhm_lorentzian():
    """Analytic FWHM = 2 · half-width = 1.0 for the same Lorentzian."""
    x = np.linspace(-5, 5, 2001)
    y = 1.0 / (1.0 + ((x - 1.0) / 0.5) ** 2)
    peak_x, peak_h, fwhm = fwhm_in_window(x, y, (-5.0, 5.0))
    assert np.isclose(peak_x, 1.0, atol=0.01)
    assert np.isclose(fwhm, 1.0, atol=0.05)


def test_fwhm_uses_abs_value():
    """A negative-going peak: |y| FWHM should match the positive twin."""
    x = np.linspace(-5, 5, 2001)
    y = -1.0 / (1.0 + (x / 0.5) ** 2)
    _, peak_h, fwhm = fwhm_in_window(x, y, (-5.0, 5.0), use_abs=True)
    assert np.isclose(peak_h, 1.0, atol=0.01)
    assert np.isclose(fwhm, 1.0, atol=0.05)


def test_fwhm_runs_off_window_returns_nan():
    """If the half-max crossing isn't found inside the window, fwhm is NaN."""
    x = np.linspace(-1, 1, 51)
    y = np.abs(x) + 1.0  # monotonically increasing from centre; no peak
    _, _, fwhm = fwhm_in_window(x, y, (-1.0, 1.0))
    assert np.isnan(fwhm)
