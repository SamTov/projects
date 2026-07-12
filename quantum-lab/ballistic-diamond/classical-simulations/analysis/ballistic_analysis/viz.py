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
    orientation: Optional[int] = None,
    bins: int = 40,
    depth_range: Optional[tuple[float, float]] = None,
    log: bool = False,
):
    """Histogram of final ion depth for one species, stratified by `by`.

    `by` is the column whose unique values become overlaid histograms
    (typical choices: "angle_deg", "T_K", "E_keV").  The other two physical
    knobs should be pinned via the keyword args, or left None to pool over.
    """
    sel = _select(summary, species=species, E_keV=E_keV, T_K=T_K,
                  orientation=orientation)
    if sel.size == 0:
        raise ValueError(
            f"No rows matched species={species}, E={E_keV}, T={T_K}, "
            f"orient={orientation}")

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
        f"{k}={v}" for k, v in
        {"E_keV": E_keV, "T_K": T_K, "orient": orientation}.items()
        if v is not None
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
    orientation: Optional[int] = None,
    statistic: str = "median",
):
    """Per-cell central depth (median or mean) +/- spread vs `x`, grouped by `group`.

    Errorbars are the inter-quartile range when statistic="median", standard
    deviation when statistic="mean".  Pin the remaining knobs via the kwargs.
    """
    pin = {"species": species, "E_keV": E_keV, "angle_deg": angle_deg,
           "T_K": T_K, "orientation": orientation}
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
    orientation: Optional[int] = None,
):
    """Fraction of ensembles where the ion ended up deeper than `threshold_A`.

    The natural threshold is something like 2-3x the SRIM mean range -- ions
    going deeper than that almost certainly channeled.

    Returns a Figure with one panel per pinned (E_keV, T_K).  X axis is angle.
    """
    sel = _select(summary, species=species, E_keV=E_keV, T_K=T_K,
                  orientation=orientation)
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

    ax.set_xlabel("Tilt off channel axis  [deg]")
    ax.set_ylabel(f"P(depth > {threshold_A:g} A)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{species.upper()} channeling fraction vs angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Vacancy / damage plots (from damage_ingest.py output)
# ----------------------------------------------------------------------

def vacancy_depth_histogram(
    metrics,
    vac_depths: Sequence[np.ndarray],
    species: str,
    *,
    by: str = "angle_deg",
    E_keV: Optional[float] = None,
    T_K: Optional[float] = None,
    angle_deg: Optional[float] = None,
    orientation: Optional[int] = None,
    bins: int = 60,
    depth_range: Optional[tuple[float, float]] = None,
    per_ion: bool = True,
    log: bool = False,
):
    """Histogram of VACANCY depths, stratified by `by`, pooled over ensembles.

    metrics/vac_depths come from damage.load_damage().  With per_ion=True the
    y-axis is vacancies per incident ion per bin (comparable to SRIM damage
    profiles); otherwise raw counts.
    """
    pin = {"species": species, "E_keV": E_keV, "T_K": T_K,
           "angle_deg": angle_deg, "orientation": orientation}
    pin.pop(by, None)
    mask = metrics["ok"] == 1
    for key, value in pin.items():
        if value is not None:
            mask &= metrics[key] == value

    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No damage rows matched {pin}")

    if depth_range is None:
        pooled_max = max(
            (float(vac_depths[i].max()) for i in idx if len(vac_depths[i])),
            default=1.0,
        )
        depth_range = (0.0, pooled_max)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for level in np.unique(metrics[by][mask]):
        rows = idx[metrics[by][idx] == level]
        pooled = np.concatenate(
            [vac_depths[i] for i in rows] or [np.zeros(0)]
        )
        weights = None
        if per_ion and len(pooled):
            weights = np.full(len(pooled), 1.0 / len(rows))
        ax.hist(pooled, bins=bins, range=depth_range, histtype="step",
                linewidth=2, label=f"{by}={level:g}", weights=weights)

    ax.set_xlabel("Vacancy depth below surface  [A]")
    ax.set_ylabel("Vacancies / ion / bin" if per_ion else "Vacancy count")
    if log:
        ax.set_yscale("log")
    pinned = " | ".join(f"{k}={v}" for k, v in pin.items()
                        if k != "species" and v is not None)
    ax.set_title(f"{species.upper()} vacancy depth profile  ({pinned or 'pooled'})")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------

def summarize_cells(summary) -> dict:
    """Return per-cell statistics as a dict of lists, easily turned into a DF.

    `n` counts implanted ions (depth stats are over these); `n_total`,
    `n_transmitted`, `n_other` expose the punch-through / failure budget --
    a large transmitted count means the depth distribution is right-censored
    by the slab thickness in that cell.

        import pandas as pd
        pd.DataFrame(summarize_cells(summary))
    """
    out = {k: [] for k in
           ("species", "orientation", "E_keV", "angle_deg", "T_K",
            "n", "n_total", "n_transmitted", "n_other",
            "depth_mean", "depth_median", "depth_std",
            "depth_p10", "depth_p90", "ke_final_mean_eV")}

    names = summary.dtype.names or ()
    has_status = "status" in names
    has_orient = "orientation" in names
    for sp in np.unique(summary["species"]):
        sA = summary[summary["species"] == sp]
        orients = np.unique(sA["orientation"]) if has_orient else [110]
        for O in orients:
            s0 = sA[sA["orientation"] == O] if has_orient else sA
            for E in np.unique(s0["E_keV"]):
                s1 = s0[s0["E_keV"] == E]
                for A in np.unique(s1["angle_deg"]):
                    s2 = s1[s1["angle_deg"] == A]
                    for T in np.unique(s2["T_K"]):
                        cell = s2[s2["T_K"] == T]
                        imp = cell[cell["ok"] == 1]
                        n_trans = int((cell["status"] == "transmitted").sum()) if has_status else 0
                        out["species"].append(str(sp))
                        out["orientation"].append(int(O))
                        out["E_keV"].append(float(E))
                        out["angle_deg"].append(float(A))
                        out["T_K"].append(float(T))
                        out["n"].append(int(imp.size))
                        out["n_total"].append(int(cell.size))
                        out["n_transmitted"].append(n_trans)
                        out["n_other"].append(int(cell.size - imp.size - n_trans))
                        if imp.size:
                            out["depth_mean"].append(float(np.nanmean(imp["depth"])))
                            out["depth_median"].append(float(np.nanmedian(imp["depth"])))
                            out["depth_std"].append(float(np.nanstd(imp["depth"])))
                            out["depth_p10"].append(float(np.nanpercentile(imp["depth"], 10)))
                            out["depth_p90"].append(float(np.nanpercentile(imp["depth"], 90)))
                            out["ke_final_mean_eV"].append(float(np.nanmean(imp["ke_eV"])))
                        else:
                            for k in ("depth_mean", "depth_median", "depth_std",
                                      "depth_p10", "depth_p90", "ke_final_mean_eV"):
                                out[k].append(float("nan"))
    return out
