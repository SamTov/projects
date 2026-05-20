"""Plotting helpers for the ballistic-diamond sweep summary.

All functions take a structured array `summary` (as returned by load_summary)
and return a matplotlib Figure.  Cell selection is by exact match on
(species, E_keV, angle_deg, T_K).
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _select(summary, **filters):
    mask = summary["ok"] == 1
    for key, value in filters.items():
        if value is None:
            continue
        mask &= summary[key] == value
    return summary[mask]


# ----------------------------------------------------------------------
# Depth distributions
# ----------------------------------------------------------------------

def depth_histogram(
    summary,
    species: str,
    *,
    by: str = "angle_deg",
    E_keV: Optional[float] = None,
    T_K: Optional[float] = None,
    bins: int = 40,
    depth_range: Optional[tuple[float, float]] = None,
    log: bool = False,
):
    """Histogram of final ion depth for one species, stratified by `by`.

    `by` is the column whose unique values become overlaid histograms
    (typical choices: "angle_deg", "T_K", "E_keV").  The other two physical
    knobs should be pinned via the keyword args, or left None to pool over.
    """
    sel = _select(summary, species=species, E_keV=E_keV, T_K=T_K)
    if sel.size == 0:
        raise ValueError(f"No rows matched species={species}, E={E_keV}, T={T_K}")

    if depth_range is None:
        depth_range = (float(np.nanmin(sel["depth"])), float(np.nanmax(sel["depth"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for level in np.unique(sel[by]):
        sub = sel[sel[by] == level]
        ax.hist(
            sub["depth"], bins=bins, range=depth_range,
            histtype="step", linewidth=2, label=f"{by}={level:g}",
            density=True,
        )

    ax.set_xlabel("Ion depth below top surface  [A]")
    ax.set_ylabel("Probability density")
    if log:
        ax.set_yscale("log")
    pinned = " | ".join(
        f"{k}={v}" for k, v in {"E_keV": E_keV, "T_K": T_K}.items() if v is not None
    )
    ax.set_title(f"{species.upper()} depth distribution  ({pinned or 'pooled'})")
    ax.legend()
    fig.tight_layout()
    return fig


def depth_vs_parameter(
    summary,
    species: str,
    x: str = "angle_deg",
    *,
    group: Optional[str] = "T_K",
    E_keV: Optional[float] = None,
    T_K: Optional[float] = None,
    angle_deg: Optional[float] = None,
    statistic: str = "median",
):
    """Per-cell central depth (median or mean) +/- spread vs `x`, grouped by `group`.

    Errorbars are the inter-quartile range when statistic="median", standard
    deviation when statistic="mean".  Pin the remaining knobs via the kwargs.
    """
    pin = {"species": species, "E_keV": E_keV, "angle_deg": angle_deg, "T_K": T_K}
    pin.pop(x, None)
    if group is not None:
        pin.pop(group, None)
    sel = _select(summary, **pin)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = [None] if group is None else np.unique(sel[group])
    for g in groups:
        s = sel if group is None else sel[sel[group] == g]
        xs = np.unique(s[x])
        center = np.zeros_like(xs, dtype=float)
        lo = np.zeros_like(xs, dtype=float)
        hi = np.zeros_like(xs, dtype=float)
        for i, xv in enumerate(xs):
            d = s["depth"][s[x] == xv]
            if statistic == "median":
                center[i] = np.nanmedian(d)
                lo[i], hi[i] = np.nanpercentile(d, [25, 75])
            elif statistic == "mean":
                center[i] = np.nanmean(d)
                sd = np.nanstd(d)
                lo[i], hi[i] = center[i] - sd, center[i] + sd
            else:
                raise ValueError(f"unknown statistic={statistic!r}")
        label = None if group is None else f"{group}={g:g}"
        ax.errorbar(xs, center, yerr=[center - lo, hi - center],
                    marker="o", capsize=3, label=label)

    ax.set_xlabel(x)
    ax.set_ylabel(f"{statistic} depth  [A]")
    pinned = " | ".join(
        f"{k}={v}" for k, v in pin.items() if k != "species" and v is not None
    )
    ax.set_title(f"{species.upper()} {statistic} depth vs {x}  ({pinned or 'pooled'})")
    if group is not None:
        ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Channeling fraction
# ----------------------------------------------------------------------

def channeling_fraction(
    summary,
    species: str,
    threshold_A: float,
    *,
    E_keV: Optional[float] = None,
    T_K: Optional[float] = None,
):
    """Fraction of ensembles where the ion ended up deeper than `threshold_A`.

    The natural threshold is something like 2-3x the SRIM mean range -- ions
    going deeper than that almost certainly channeled.

    Returns a Figure with one panel per pinned (E_keV, T_K).  X axis is angle.
    """
    sel = _select(summary, species=species, E_keV=E_keV, T_K=T_K)
    angles = np.unique(sel["angle_deg"])
    energies = np.unique(sel["E_keV"])
    temps = np.unique(sel["T_K"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for E in energies:
        for T in temps:
            mask = (sel["E_keV"] == E) & (sel["T_K"] == T)
            if not mask.any():
                continue
            sub = sel[mask]
            frac = np.array([
                (sub["depth"][sub["angle_deg"] == a] > threshold_A).mean()
                for a in angles
            ])
            ax.plot(angles, frac, marker="o", label=f"E={E:g} keV, T={T:g} K")

    ax.set_xlabel("Tilt off [-110] channel  [deg]")
    ax.set_ylabel(f"P(depth > {threshold_A:g} A)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{species.upper()} channeling fraction vs angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------

def summarize_cells(summary) -> dict:
    """Return per-cell statistics as a dict of lists, easily turned into a DF.

    Useful for quick text-printing or pandas conversion:
        import pandas as pd
        pd.DataFrame(summarize_cells(summary))
    """
    ok = summary[summary["ok"] == 1]
    out = {k: [] for k in
           ("species", "E_keV", "angle_deg", "T_K",
            "n", "depth_mean", "depth_median", "depth_std",
            "depth_p10", "depth_p90", "ke_final_mean_eV")}

    species_unique = np.unique(ok["species"])
    for sp in species_unique:
        s0 = ok[ok["species"] == sp]
        for E in np.unique(s0["E_keV"]):
            s1 = s0[s0["E_keV"] == E]
            for A in np.unique(s1["angle_deg"]):
                s2 = s1[s1["angle_deg"] == A]
                for T in np.unique(s2["T_K"]):
                    s3 = s2[s2["T_K"] == T]
                    out["species"].append(str(sp))
                    out["E_keV"].append(float(E))
                    out["angle_deg"].append(float(A))
                    out["T_K"].append(float(T))
                    out["n"].append(int(s3.size))
                    out["depth_mean"].append(float(np.nanmean(s3["depth"])))
                    out["depth_median"].append(float(np.nanmedian(s3["depth"])))
                    out["depth_std"].append(float(np.nanstd(s3["depth"])))
                    out["depth_p10"].append(float(np.nanpercentile(s3["depth"], 10)))
                    out["depth_p90"].append(float(np.nanpercentile(s3["depth"], 90)))
                    out["ke_final_mean_eV"].append(float(np.nanmean(s3["ke_eV"])))
    return out
