"""Experiment 5 post-processing: Δ^σ(T) comparison + cross-block spectrum."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics.plotting import apply_style


def _load(path: Path) -> list[dict]:
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _agg(rows, key):
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


def fig_delta_vs_T(rows, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    for key, label, marker in [
        ("Delta_sigma_matrix", r"$\Delta^\sigma$ (matrix blocks)", "o"),
        ("Delta_sigma_from_C_charges", r"$\Delta^\sigma$ ($z^T C z$)", "^"),
        ("Delta_sigma_from_time_GK", r"$\Delta^\sigma$ (time-domain GK)", "s"),
    ]:
        Ts, means, stds = _agg(rows, key)
        ax.errorbar(Ts, means, yerr=stds, marker=marker, capsize=2, label=label)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$\Delta^\sigma$")
    ax.legend(frameon=False)
    fig.savefig(outfile)
    plt.close(fig)


def fig_rdf_and_coordination(rows, runs_dir: Path, outfile: Path) -> None:
    """Stacked panels: g_NaCl(r) at low/high T, plus coordination(T)."""
    if not rows:
        return
    Ts_all = sorted({float(r["temperature"]) for r in rows})
    T_lo, T_hi = Ts_all[0], Ts_all[-1]
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), constrained_layout=True)

    for target_T, color in [(T_lo, "C0"), (T_hi, "C3")]:
        rs = []
        gs = []
        for r in rows:
            if float(r["temperature"]) != target_T:
                continue
            tag = f"temperature={target_T:g}_seed={int(r['seed'])}"
            r_path = runs_dir / tag / "rdf_NaCl_r.npy"
            g_path = runs_dir / tag / "rdf_NaCl_g.npy"
            if r_path.is_file() and g_path.is_file():
                rs.append(np.load(r_path))
                gs.append(np.load(g_path))
        if rs:
            g_mean = np.mean(np.stack(gs, axis=0), axis=0)
            axes[0].plot(rs[0], g_mean, label=f"T = {target_T:g} K", color=color)
    axes[0].set_xlabel("r (Å)")
    axes[0].set_ylabel(r"$g_{\mathrm{NaCl}}(r)$")
    axes[0].legend(frameon=False)

    Ts, coord, coord_err = _agg(rows, "coord_number_NaCl")
    axes[1].errorbar(Ts, coord, yerr=coord_err, marker="o", capsize=2, color="C2")
    axes[1].set_xlabel("T (K)")
    axes[1].set_ylabel(r"$N_{\mathrm{NaCl}}$ (first shell)")

    fig.savefig(outfile)
    plt.close(fig)


def fig_svd_contact_correlation(rows, outfile: Path) -> None:
    """Spearman ρ between leading u_1 and per-Na contact degree, vs T."""
    Ts, rho, rho_err = _agg(rows, "svd_contact_spearman")
    if Ts.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.errorbar(Ts, rho, yerr=rho_err, marker="o", capsize=2, color="C4")
    ax.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"Spearman $\rho(|u_1|,\,N_{\mathrm{NaCl}}^{(i)})$")
    fig.savefig(outfile)
    plt.close(fig)


def fig_crossblock_spectrum(rows, runs_dir: Path, outfile: Path) -> None:
    """Plot the singular-value spectrum of the Na–Cl cross block at the
    lowest simulated T (tightest ion correlations)."""
    if not rows:
        return
    T_target = min(float(r["temperature"]) for r in rows)
    cohort = [r for r in rows if float(r["temperature"]) == T_target]
    svs = []
    for r in cohort:
        tag = f"temperature={T_target:g}_seed={int(r['seed'])}"
        p = runs_dir / tag / "cross_block_svd.npy"
        if p.is_file():
            svs.append(np.load(p))
    if not svs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    for i, s in enumerate(svs):
        ax.plot(np.arange(1, s.size + 1), s, marker="o", markersize=3,
                alpha=0.6, label=f"seed {int(cohort[i]['seed'])}")
    ax.set_yscale("log")
    ax.set_xlabel("index k")
    ax.set_ylabel(r"singular value $\sigma_k$")
    ax.set_title(f"cross-block spectrum, T = {T_target:g} K")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(outfile)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    apply_style()
    rows = _load(args.output_dir / "salt_summary.csv")
    figs = args.output_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_delta_vs_T(rows, figs / "fig_salt_delta_vs_T.pdf")
    fig_crossblock_spectrum(rows, args.output_dir / "runs", figs / "fig_salt_crossblock_spectrum.pdf")
    fig_rdf_and_coordination(rows, args.output_dir / "runs", figs / "fig_salt_rdf_coordination.pdf")
    fig_svd_contact_correlation(rows, figs / "fig_salt_svd_contact_corr.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
