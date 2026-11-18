"""Manuscript figures for sections 4.2 and 4.3 from the copied result tables.

Usage: python paper_figures.py <results_dir> <figures_dir>
Expects results/02_bond_addition_long/bond_summary.csv and
results/03_lj_phases/lj_summary.csv (as written by the experiment drivers).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics.plotting import apply_style


def wishart_entropy_table(N: int, qs: np.ndarray, reps: int = 4, seed: int = 1) -> np.ndarray:
    """S/ln N of a white Wishart matrix with N columns and N/q rows."""
    rng = np.random.default_rng(seed)
    out = []
    for q in qs:
        vals = []
        for _ in range(reps):
            X = rng.standard_normal((int(round(N / q)), N))
            mu = np.linalg.eigvalsh(X.T @ X)
            mu = mu[mu > 1e-12 * mu.max()]
            p = mu / mu.sum()
            vals.append(-(p * np.log(p)).sum() / np.log(N))
        out.append(np.mean(vals))
    return np.array(out)


def lj_figure(rows, outfile: Path, N: int = 500) -> list[dict]:
    qs = np.geomspace(0.07, 1.6, 18)
    tab = wishart_entropy_table(N, qs)
    null = lambda q: np.interp(np.log(q), np.log(qs), tab)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["init"], float(r["temperature"]))].append(r)
    branches = {"fcc": ("C0", "o", "crystal start"), "dense": ("C1", "s", "dense start"),
                "dilute": ("C2", "^", "dilute start")}
    table = []
    fig, axes = plt.subplots(3, 1, figsize=(5, 8.5), sharex=True, constrained_layout=True)
    for b, (c, m, lbl) in branches.items():
        Ts = sorted(T for (bb, T) in groups if bb == b)
        rho, drho, ex, dex, D, dD, q = [], [], [], [], [], [], []
        for T in Ts:
            rs = groups[(b, T)]
            f = lambda k: np.array([float(r[k]) for r in rs])
            qq = N / f("T_eff")
            e = np.array([null(x) for x in qq]) - f("S_over_logN")
            rho.append(f("density").mean()); drho.append(f("density").std())
            ex.append(e.mean()); dex.append(e.std())
            D.append(f("diffusion").mean() * 1e4); dD.append(f("diffusion").std() * 1e4)
            q.append(qq.mean())
            table.append({"init": b, "T": T, "rho": rho[-1], "excess": ex[-1], "excess_sd": dex[-1],
                          "D": D[-1], "q": q[-1], "S": f("S_over_logN").mean()})
        Ts, q = np.array(Ts), np.array(q)
        axes[0].errorbar(Ts, rho, yerr=drho, color=c, marker=m, ms=4, capsize=2, label=lbl)
        # The moment-matched null is only reliable well below q = 1.
        ok = q < 0.6
        axes[1].errorbar(Ts[ok], np.array(ex)[ok], yerr=np.array(dex)[ok], color=c, marker=m, ms=4, capsize=2)
        axes[2].errorbar(Ts, D, yerr=dD, color=c, marker=m, ms=4, capsize=2)
    axes[0].set_ylabel(r"density (g cm$^{-3}$)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel(r"$[S_{\mathrm{MP}} - S(C)] / \ln |\mathcal{N}|$")
    axes[1].axhline(0.0, color="0.5", lw=0.8, ls="--")
    axes[2].set_ylabel(r"$\mathrm{Tr}(C)/|\mathcal{N}|$ ($10^{-4}$ Å$^2$ fs$^{-1}$)")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("T (K)")
    for ax, l in zip(axes, "abc"):
        ax.text(0.02, 0.92, f"({l})", transform=ax.transAxes)
    fig.savefig(outfile)
    plt.close(fig)
    return table


def bond_long_figure(rows, counts, outfile: Path) -> None:
    ks = sorted({int(r["k_bonds"]) for r in rows})
    agg = lambda k, key: np.array([float(r[key]) for r in rows if int(r["k_bonds"]) == k])
    nc = np.array([agg(k, "n_constraints").mean() for k in ks])
    # Sub-bulk counts against the k = 0 edges (bond_mode_counts.csv), minus
    # the zero mode from total-momentum conservation.
    nb = [np.array([float(c["n_below"]) for c in counts if int(c["k_bonds"]) == k]) - 1 for k in ks]
    ov = [agg(k, "overlap_lowest") for k in ks]
    N = 216
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    axes[0].errorbar(nc, [x.mean() for x in nb], yerr=[x.std() for x in nb], marker="o", capsize=3, color="C0",
                     label="below lower edge")
    axes[0].plot([0, nc.max()], [0, nc.max()], ls="--", color="0.5", lw=0.8, label="one per constraint")
    axes[0].set_xlabel("independent bond constraints")
    axes[0].set_ylabel("sub-bulk modes (excl. momentum)")
    axes[0].legend(frameon=False)
    keep = nc > 0
    axes[1].errorbar(nc[keep], [ov[i].mean() for i in np.where(keep)[0]],
                     yerr=[ov[i].std() for i in np.where(keep)[0]], marker="o", capsize=3, color="C0", label="lowest modes")
    axes[1].plot(nc[keep], nc[keep] / N, ls=":", color="0.4", label="random modes")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("independent bond constraints")
    axes[1].set_ylabel("overlap with bond space")
    axes[1].legend(frameon=False)
    fig.savefig(outfile)
    plt.close(fig)


def main() -> int:
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    apply_style()
    lj = list(csv.DictReader(open(res / "03_lj_phases" / "lj_summary.csv")))
    table = lj_figure(lj, figs / "fig_lj_isobar.pdf")
    with open(res / "03_lj_phases" / "entropy_excess.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    bl = list(csv.DictReader(open(res / "02_bond_addition_long" / "bond_summary.csv")))
    bc = list(csv.DictReader(open(res / "02_bond_addition_long" / "bond_mode_counts.csv")))
    bond_long_figure(bl, bc, figs / "fig_bond_long.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
