"""Post-processing for Experiment 2. Reads bond_summary.csv and per-run
artifacts to produce the three spec figures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics import load_C
from rmt_dynamics.plotting import apply_style, plot_heatmap, plot_spectrum


def _load_summary(path: Path) -> list[dict]:
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def fig_bond_spectrum(rows, runs_dir: Path, outfile: Path) -> None:
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
        if lam_plus is not None:
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
    fig_bond_spectrum(rows, args.output_dir / "runs", figs / "fig_bond_spectrum.pdf")
    fig_bond_entropy(rows, figs / "fig_bond_entropy.pdf")
    fig_bond_heatmap(rows, args.output_dir / "runs", figs / "fig_bond_heatmap.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
