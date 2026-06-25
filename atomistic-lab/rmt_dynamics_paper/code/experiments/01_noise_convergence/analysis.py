"""Post-processing for Experiment 1 — reads ``noise_summary.csv`` and
per-run ``eigvals.npy`` files, produces figures and derived CSVs.

Outputs:
- ``figures/fig_noise_eigdensity.pdf`` — per-N eigenvalue histogram vs MP.
- ``figures/fig_noise_entropy_convergence.pdf`` — S(C)/log(N) vs T_traj.
- ``figures/fig_noise_deviation_fits.pdf`` — |log(N) - S| vs T_traj on
  log-log axes with the fitted power law per N.
- ``convergence_exponents.csv`` — fitted exponent + uncertainty per N.

Run after the SLURM array finishes::

    python analysis.py --output-dir outputs
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics.plotting import apply_style, plot_spectrum


def _load_summary(path: Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def fig_eigdensity(rows, runs_dir: Path, outfile: Path) -> None:
    """One panel per N at the largest T_traj, overlaying MP."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    by_N: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_N.setdefault(int(row["n_atoms"]), []).append(row)
    Ns = sorted(by_N)
    for ax, N in zip(axes, Ns):
        cohort = [r for r in by_N[N] if int(r["n_prod"]) == max(int(x["n_prod"]) for x in by_N[N])]
        eigs = []
        for r in cohort:
            tag = f"n_atoms={N}_n_prod={int(r['n_prod'])}_seed={int(r['seed'])}"
            eig_path = runs_dir / tag / "eigvals.npy"
            if eig_path.is_file():
                eigs.append(np.load(eig_path))
        if not eigs:
            ax.set_title(f"N={N} (no data)")
            continue
        pooled = np.concatenate(eigs)
        mp = {"N": N, "T_eff": float(cohort[0]["T_eff"]), "sigma2": float(cohort[0]["sigma2"])}
        plot_spectrum(ax, pooled, mp_params=mp, bins=60)
        ax.set_title(f"N = {N}")
    fig.savefig(outfile)
    plt.close(fig)


def fig_entropy_convergence(rows, outfile: Path) -> None:
    """S(C)/log(N) vs T_traj, one series per N, error bars over seeds."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    by_N: dict[int, dict[int, list[float]]] = {}
    for row in rows:
        N = int(row["n_atoms"])
        T_traj = int(row["n_prod"])
        by_N.setdefault(N, {}).setdefault(T_traj, []).append(float(row["S_over_logN"]))
    for N in sorted(by_N):
        xs = sorted(by_N[N])
        means = [float(np.mean(by_N[N][t])) for t in xs]
        errs = [float(np.std(by_N[N][t])) for t in xs]
        ax.errorbar(xs, means, yerr=errs, marker="o", capsize=3, label=f"N = {N}")
    ax.set_xscale("log")
    ax.set_xlabel(r"production steps $T_{\mathrm{traj}}$")
    ax.set_ylabel(r"$S(C)\,/\,\log N$")
    ax.axhline(1.0, color="k", linewidth=0.6, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(outfile)
    plt.close(fig)


def fit_convergence_exponents(rows) -> list[dict[str, Any]]:
    """Fit `|log(N) - S(C)|` ~ A · T_traj^α per N.

    Uses a weighted least-squares fit in log-log space; weights come from the
    per-cell seed-to-seed standard deviation of `log10(deviation)`. Returns a
    list of dicts (one per N) with the fitted exponent, intercept, uncertainty,
    and reduced chi-square. Cells where the deviation is non-positive (would
    overshoot log N) are dropped from the fit.
    """
    by_N: dict[int, dict[int, list[float]]] = {}
    for r in rows:
        N = int(r["n_atoms"])
        T = int(r["n_prod"])
        dev = float(np.log(N)) - float(r["S"])
        if dev > 0:
            by_N.setdefault(N, {}).setdefault(T, []).append(dev)

    fits: list[dict[str, Any]] = []
    for N, by_T in sorted(by_N.items()):
        Ts = sorted(by_T)
        if len(Ts) < 2:
            continue  # need at least two T_traj to fit a slope
        log_T = np.array([np.log10(t) for t in Ts])
        log_dev_mean = np.array([np.log10(np.mean(by_T[t])) for t in Ts])
        # Seed scatter on the log10 of deviation. With one seed only, fall back
        # to a small constant so the fit remains well-conditioned.
        log_dev_err = np.array([
            float(np.std(np.log10(by_T[t]))) if len(by_T[t]) > 1 else 0.1
            for t in Ts
        ])
        # numpy.polyfit weights are 1/sigma.
        w = 1.0 / np.where(log_dev_err > 0, log_dev_err, 0.1)
        try:
            coefs, cov = np.polyfit(log_T, log_dev_mean, 1, w=w, cov=True)
        except (np.linalg.LinAlgError, ValueError):
            continue
        slope, intercept = float(coefs[0]), float(coefs[1])
        slope_err = float(np.sqrt(cov[0, 0]))
        # Reduced chi-square as a fit-quality flag.
        pred = slope * log_T + intercept
        resid = (log_dev_mean - pred) / np.where(log_dev_err > 0, log_dev_err, 0.1)
        dof = max(1, len(Ts) - 2)
        chi2_red = float(np.sum(resid ** 2) / dof)
        fits.append({
            "n_atoms": N,
            "exponent": slope,
            "exponent_err": slope_err,
            "intercept": intercept,
            "n_T_traj": len(Ts),
            "chi2_reduced": chi2_red,
        })
    return fits


def fig_deviation_fits(rows, fits, outfile: Path) -> None:
    """Plot |log(N) − S| vs T_traj with the fitted power law overlayed per N."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    by_N: dict[int, dict[int, list[float]]] = {}
    for r in rows:
        N = int(r["n_atoms"])
        T = int(r["n_prod"])
        dev = float(np.log(N)) - float(r["S"])
        if dev > 0:
            by_N.setdefault(N, {}).setdefault(T, []).append(dev)

    fit_by_N = {f["n_atoms"]: f for f in fits}
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(by_N)))

    for color, (N, by_T) in zip(colors, sorted(by_N.items())):
        Ts = sorted(by_T)
        means = [float(np.mean(by_T[t])) for t in Ts]
        errs = [float(np.std(by_T[t])) if len(by_T[t]) > 1 else 0.0 for t in Ts]
        ax.errorbar(Ts, means, yerr=errs, marker="o", capsize=3,
                    color=color, label=f"N = {N}", linestyle="none")
        if N in fit_by_N:
            f = fit_by_N[N]
            xs = np.logspace(np.log10(min(Ts)), np.log10(max(Ts)), 100)
            ys = 10.0 ** (f["intercept"]) * xs ** (f["exponent"])
            ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1.0,
                    label=rf"slope = {f['exponent']:.2f} ± {f['exponent_err']:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"production steps $T_{\mathrm{traj}}$")
    ax.set_ylabel(r"$|\log N - S(C)|$")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(outfile)
    plt.close(fig)


def write_exponent_csv(fits, outfile: Path) -> None:
    if not fits:
        return
    keys = list(fits[0].keys())
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in fits:
            w.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    apply_style()
    summary_path = args.output_dir / "noise_summary.csv"
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not summary_path.is_file():
        raise SystemExit(f"{summary_path} not found — run the sweep first")
    rows = _load_summary(summary_path)

    fig_eigdensity(rows, args.output_dir / "runs", figures_dir / "fig_noise_eigdensity.pdf")
    fig_entropy_convergence(rows, figures_dir / "fig_noise_entropy_convergence.pdf")

    fits = fit_convergence_exponents(rows)
    write_exponent_csv(fits, args.output_dir / "convergence_exponents.csv")
    fig_deviation_fits(rows, fits, figures_dir / "fig_noise_deviation_fits.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
