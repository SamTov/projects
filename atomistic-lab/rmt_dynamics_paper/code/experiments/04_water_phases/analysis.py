"""Experiment 4 post-processing: entropy vs T (O-only and all-atom) and
eigenvector / H-bond overlap scatter."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics import eigendecomposition, load_C, participation_ratio
from rmt_dynamics.plotting import apply_style


def _load(path: Path) -> list[dict]:
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _aggregate(rows, key):
    out = {}
    for r in rows:
        T = float(r["temperature"])
        v = float(r[key])
        if not np.isnan(v):
            out.setdefault(T, []).append(v)
    Ts = sorted(out)
    return (
        np.array(Ts),
        np.array([np.mean(out[T]) for T in Ts]),
        np.array([np.std(out[T]) for T in Ts]),
    )


def fig_entropy(rows, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    for key, label, marker in [
        ("S_oxygens", r"$S_O(C)$", "o"),
        ("S_all", r"$S_\mathrm{all}(C)$", "s"),
    ]:
        Ts, means, stds = _aggregate(rows, key)
        ax.errorbar(Ts, means, yerr=stds, marker=marker, capsize=2, label=label)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("entropy")
    ax.legend(frameon=False)
    fig.savefig(outfile)
    plt.close(fig)


def fig_hbond_overlap(runs_dir: Path, outfile: Path, target_T: float = 300.0) -> None:
    """Scatter top-eigenvector localisation against the per-oxygen H-bond
    degree from the Luzar–Chandler graph at T = `target_T`.

    Each seed produces one scatter cloud; the per-seed Spearman ρ is shown
    in the legend so the reader can see whether the correlation holds across
    independent runs.
    """
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    pooled_amp: list[np.ndarray] = []
    pooled_deg: list[np.ndarray] = []
    legend_lines = []
    for run in sorted(runs_dir.glob(f"temperature={target_T:g}*_seed=*")):
        C_path = run / "C_oxygens.npz"
        deg_path = run / "hbond_degree_mean.npy"
        eigvec_path = run / "eigvecs_oxygens.npy"
        if not (C_path.is_file() and deg_path.is_file() and eigvec_path.is_file()):
            continue
        Phi = np.load(eigvec_path)
        degree = np.load(deg_path)
        top_k = min(5, Phi.shape[1])
        amp = np.mean(np.abs(Phi[:, -top_k:]), axis=1)
        ax.scatter(degree, amp, s=10, alpha=0.5)
        pooled_amp.append(amp)
        pooled_deg.append(degree)
        seed = run.name.split("seed=")[-1]
        if np.std(amp) > 0 and np.std(degree) > 0:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(amp, degree)
            legend_lines.append(f"seed {seed}: ρ = {rho:.2f}")
        else:
            legend_lines.append(f"seed {seed}: ρ undefined")
    ax.set_xlabel("H-bond degree (Luzar–Chandler)")
    ax.set_ylabel(r"$\langle |\phi_{ik}| \rangle_{k \in \mathrm{top}\,5}$")
    ax.set_title(f"T = {target_T:g} K")
    if legend_lines:
        ax.legend(legend_lines, frameon=False, fontsize=7, loc="best",
                  handlelength=0, handletextpad=0)
    fig.savefig(outfile)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    apply_style()
    rows = _load(args.output_dir / "water_summary.csv")
    figs = args.output_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_entropy(rows, figs / "fig_water_entropy.pdf")
    fig_hbond_overlap(args.output_dir / "runs", figs / "fig_water_hbond_overlap.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
