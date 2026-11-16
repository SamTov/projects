"""Post-processing for Experiment 2. Reads bond_summary.csv and per-run
artifacts to produce the three spec figures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics import load_C, mp_edges
from rmt_dynamics.plotting import apply_style, plot_heatmap, plot_spectrum


def _load_summary(path: Path) -> list[dict]:
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def baseline_edges(rows, runs_dir: Path) -> tuple[float, float] | None:
    """MP bulk edges taken from the bond-free (k = 0) liquid.

    The per-run edges use sigma2 = trace/N, which falls as atoms are bonded
    (a bonded atom diffuses more slowly), so the unbonded liquid is the
    reference against which modes leaving the bulk are counted.
    """
    base = [r for r in rows if int(r["k_bonds"]) == 0 and r.get("sigma2")]
    if not base:
        return None
    sigma2 = float(np.mean([float(r["sigma2"]) for r in base]))
    T_eff = float(np.mean([float(r["T_eff"]) for r in base]))
    p = runs_dir / f"k_bonds=0_seed={int(base[0]['seed'])}" / "eigvals.npy"
    if not p.is_file():
        return None
    return mp_edges(int(np.load(p).size), T_eff, sigma2)


def count_modes(rows, runs_dir: Path, edges) -> list[dict]:
    """Recount sub- and super-bulk modes per run against the k = 0 edges."""
    out = []
    for r in rows:
        k, seed = int(r["k_bonds"]), int(r["seed"])
        p = runs_dir / f"k_bonds={k}_seed={seed}" / "eigvals.npy"
        if not p.is_file():
            continue
        mu = np.load(p)
        out.append({
            "k_bonds": k,
            "seed": seed,
            "n_constraints": int(float(r.get("n_constraints") or 0)),
            "n_below": int((mu < edges[0]).sum()),
            "n_above": int((mu > edges[1]).sum()),
            "overlap_lowest": float(r.get("overlap_lowest") or "nan"),
        })
    return out


def fig_bond_modes(counts, outfile: Path) -> None:
    """Modes below the bulk vs independent bond constraints, and the overlap
    of the lowest modes with the bonded relative coordinates."""
    ks = sorted({c["k_bonds"] for c in counts})
    def agg(key):
        vals = [[c[key] for c in counts if c["k_bonds"] == k] for k in ks]
        return np.array([np.nanmean(v) for v in vals]), np.array([np.nanstd(v) for v in vals])
    nc, _ = agg("n_constraints")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for key, lbl, c in [("n_below", "below lower edge", "C0"), ("n_above", "above upper edge", "C3")]:
        m, s = agg(key)
        axes[0].errorbar(nc, m, yerr=s, marker="o", capsize=3, color=c, label=lbl)
    lim = max(1.0, float(nc.max()))
    axes[0].plot([0, lim], [0, lim], color="0.5", linestyle="--", linewidth=0.8, label="one per constraint")
    axes[0].set_xlabel("independent bond constraints")
    axes[0].set_ylabel("modes outside the bulk")
    axes[0].legend(frameon=False)
    m, s = agg("overlap_lowest")
    keep = nc > 0
    axes[1].errorbar(nc[keep], m[keep], yerr=s[keep], marker="o", capsize=3, color="C0")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("independent bond constraints")
    axes[1].set_ylabel("overlap of lowest modes with bond space")
    fig.savefig(outfile)
    plt.close(fig)


def fig_bond_spectrum(rows, runs_dir: Path, outfile: Path, edges=None) -> None:
    ks = sorted({int(r["k_bonds"]) for r in rows})
    fig, axes = plt.subplots(1, len(ks), figsize=(3 * len(ks), 3), constrained_layout=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        cohort = [r for r in rows if int(r["k_bonds"]) == k]
        eigs = []
        lam_plus = None
        for r in cohort:
            tag = f"k_bonds={k}_seed={int(r['seed'])}"
            p = runs_dir / tag / "eigvals.npy"
            if p.is_file():
                eigs.append(np.load(p))
            lam_plus = float(r["lambda_plus_MP"])
        if not eigs:
            ax.set_title(f"k = {k} (no data)")
            continue
        pooled = np.concatenate(eigs)
        plot_spectrum(ax, pooled, bins=60)
        if edges is not None:
            for edge in edges:
                ax.axvline(edge, color="red", linestyle="--", linewidth=0.8)
        elif lam_plus is not None:
            ax.axvline(lam_plus, color="red", linestyle="--", linewidth=0.8,
                       label=rf"$\lambda_+$ MP")
        ax.set_title(f"k = {k}")
        ax.set_yscale("log")
    fig.savefig(outfile)
    plt.close(fig)


def fig_bond_entropy(rows, outfile: Path) -> None:
    by_k: dict[int, list[float]] = {}
    for r in rows:
        by_k.setdefault(int(r["k_bonds"]), []).append(float(r["S"]))
    ks = sorted(by_k)
    means = [float(np.mean(by_k[k])) for k in ks]
    stds = [float(np.std(by_k[k])) for k in ks]
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("number of bonds k")
    ax.set_ylabel(r"$S(C)$")
    fig.savefig(outfile)
    plt.close(fig)


def fig_bond_heatmap(rows, runs_dir: Path, outfile: Path) -> None:
    ks = sorted({int(r["k_bonds"]) for r in rows})
    if not ks:
        return
    k_lo, k_hi = ks[0], ks[-1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, k in zip(axes, (k_lo, k_hi)):
        seed = min(int(r["seed"]) for r in rows if int(r["k_bonds"]) == k)
        tag = f"k_bonds={k}_seed={seed}"
        path = runs_dir / tag / "C.npz"
        if not path.is_file():
            ax.set_title(f"k = {k} (missing)")
            continue
        C, _ = load_C(path)
        im = plot_heatmap(ax, C, log=True)
        fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}|C_{ij}|$")
        ax.set_title(f"k = {k}")
    fig.savefig(outfile)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    apply_style()
    rows = _load_summary(args.output_dir / "bond_summary.csv")
    figs = args.output_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    edges = baseline_edges(rows, runs_dir)
    fig_bond_spectrum(rows, runs_dir, figs / "fig_bond_spectrum.pdf", edges=edges)
    if edges is not None:
        counts = count_modes(rows, runs_dir, edges)
        if counts:
            with open(args.output_dir / "bond_mode_counts.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(counts[0].keys()))
                w.writeheader()
                w.writerows(counts)
            fig_bond_modes(counts, figs / "fig_bond_modes.pdf")
    fig_bond_entropy(rows, figs / "fig_bond_entropy.pdf")
    fig_bond_heatmap(rows, args.output_dir / "runs", figs / "fig_bond_heatmap.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
