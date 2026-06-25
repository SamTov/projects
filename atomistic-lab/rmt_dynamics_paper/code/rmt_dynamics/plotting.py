"""Shared matplotlib styling and small plotting helpers used across experiments."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .rmt_null import mp_density, mp_edges

__all__ = ["apply_style", "plot_spectrum", "plot_heatmap"]


def apply_style() -> None:
    """Set matplotlib rcParams to a print-friendly style that matches achemso."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_spectrum(
    ax: plt.Axes,
    eigvals: np.ndarray,
    mp_params: dict[str, Any] | None = None,
    bins: int = 80,
    log_y: bool = False,
) -> None:
    """Histogram of eigenvalues with an optional MP density overlay.

    `mp_params` keys: `N`, `T_eff`, `sigma2` (defaults to 1.0).
    """
    eigvals = np.asarray(eigvals)
    ax.hist(eigvals, bins=bins, density=True, alpha=0.65, color="#1f77b4",
            edgecolor="none", label="empirical")
    if mp_params is not None:
        N = int(mp_params["N"])
        T_eff = float(mp_params["T_eff"])
        sigma2 = float(mp_params.get("sigma2", 1.0))
        lo, hi = mp_edges(N, T_eff, sigma2)
        xs = np.linspace(max(1e-12, lo * 0.9), hi * 1.05, 400)
        ax.plot(xs, mp_density(xs, N, T_eff, sigma2), color="black",
                linewidth=1.5, label="MP null")
        ax.axvline(hi, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"eigenvalue $\mu$")
    ax.set_ylabel(r"density $\rho(\mu)$")
    if log_y:
        ax.set_yscale("log")
    ax.legend(frameon=False)


def plot_heatmap(
    ax: plt.Axes,
    C: np.ndarray,
    log: bool = False,
    cmap: str = "magma",
) -> plt.cm.ScalarMappable:
    """Plot |C_ij| as a heatmap. Returns the imshow handle for adding colorbars."""
    C = np.asarray(C)
    data = np.abs(C)
    if log:
        data = np.log10(np.maximum(data, 1e-16))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xlabel("particle j")
    ax.set_ylabel("particle i")
    return im
