"""Peak location and FWHM on sampled curves.

Used by experiment analyses to quantify the width of a derivative peak
(e.g. dS/dT, dρ/dT across a phase transition). Half-max crossings are
located by linear interpolation between sample points.
"""
from __future__ import annotations

import numpy as np

__all__ = ["find_peak_in_window", "fwhm_in_window"]


def find_peak_in_window(
    x: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    use_abs: bool = True,
) -> tuple[float, float]:
    """Locate the maximum of (|y| if use_abs else y) on x ∈ [window[0], window[1]].

    Returns (peak_x, peak_height). NaN, NaN if the window contains < 1 point.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x >= window[0]) & (x <= window[1])
    if not mask.any():
        return float("nan"), float("nan")
    yw = np.abs(y[mask]) if use_abs else y[mask]
    xw = x[mask]
    idx = int(np.argmax(yw))
    return float(xw[idx]), float(yw[idx])


def fwhm_in_window(
    x: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    use_abs: bool = True,
) -> tuple[float, float, float]:
    """Full-width half-max of the peak inside `window`.

    Returns (peak_x, peak_height, fwhm). Half-max crossings are interpolated
    linearly between sample points. fwhm is NaN if either crossing is missing
    (e.g. the peak runs off the window edge).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x >= window[0]) & (x <= window[1])
    if not mask.any():
        return float("nan"), float("nan"), float("nan")
    xw = x[mask]
    yw = np.abs(y[mask]) if use_abs else y[mask]
    if xw.size < 2:
        return float(xw[0]), float(yw[0]), float("nan")
    idx = int(np.argmax(yw))
    peak_x = float(xw[idx])
    peak_h = float(yw[idx])
    half = peak_h / 2.0

    def _cross(i_from: int, step: int) -> float | None:
        """Walk from idx in the given direction until yw drops below half;
        interpolate linearly between the bracketing samples."""
        i = i_from
        while 0 <= i + step < xw.size and 0 <= i + step:
            j = i + step
            if yw[j] < half <= yw[i]:
                if yw[i] == yw[j]:
                    return float(0.5 * (xw[i] + xw[j]))
                # Linear interp between (xw[i], yw[i]) and (xw[j], yw[j]).
                frac = (yw[i] - half) / (yw[i] - yw[j])
                return float(xw[i] + frac * (xw[j] - xw[i]))
            i = j
        return None

    left = _cross(idx, -1)
    right = _cross(idx, +1)
    if left is None or right is None:
        return peak_x, peak_h, float("nan")
    return peak_x, peak_h, float(right - left)
