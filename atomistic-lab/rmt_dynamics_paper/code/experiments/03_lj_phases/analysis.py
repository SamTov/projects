"""Experiment 3 post-processing: density + entropy panels, phase spectra,
and FWHM extraction for the `S(C)` vs `ρ` transition-resolution claim.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics import fwhm_in_window
from rmt_dynamics.plotting import apply_style, plot_spectrum

# Windows in T (K) where the two first-order transitions are expected.
# Solid–liquid for argon-parameter LJ is ~20–25 K; liquid–vapor at P = 1 atm
# is ~85–90 K. Widen by a few K on either side so peak-finding has slack.
TRANSITION_WINDOWS = {
    "melt": (10.0, 50.0),
    "boil": (70.0, 110.0),
}


def _load_summary(path: Path) -> list[dict]:
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
    means = np.array([np.mean(out[T]) for T in Ts])
    stds = np.array([np.std(out[T]) for T in Ts])
    return np.array(Ts), means, stds


def fig_density_entropy(rows, outfile: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True, constrained_layout=True)
    Ts, rho, rho_err = _aggregate(rows, "density")
    _, S, S_err = _aggregate(rows, "S")

    axes[0].errorbar(Ts, rho, yerr=rho_err, marker="o", capsize=2, color="C0")
    axes[0].set_ylabel(r"density (g/cm$^3$)")

    axes[1].errorbar(Ts, S, yerr=S_err, marker="o", capsize=2, color="C1")
    axes[1].set_ylabel(r"$S(C)$")

    # Numerical derivatives via central differences.
    def ddx(y):
        d = np.gradient(y, Ts)
        return d
    axes[2].plot(Ts, ddx(rho), marker="o", color="C0", label=r"$d\rho/dT$")
    axes[2].plot(Ts, ddx(S), marker="s", color="C1", label=r"$dS/dT$")
    axes[2].legend(frameon=False)
    axes[2].set_xlabel("T (K)")
    axes[2].set_ylabel("derivative")

    fig.savefig(outfile)
    plt.close(fig)


def fig_spectra_by_phase(rows, runs_dir: Path, outfile: Path) -> None:
    """Representative T in each phase (solid, liquid, gas) — uses 20, 60, 100 K."""
    targets = [20.0, 60.0, 100.0]
    labels = ["solid (T=20 K)", "liquid (T=60 K)", "vapor (T=100 K)"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for ax, T, lbl in zip(axes, targets, labels):
        cohort = [r for r in rows if float(r["temperature"]) == T]
        eigs = []
        for r in cohort:
            tag = f"temperature={T:g}_seed={int(r['seed'])}"
            p = runs_dir / tag / "eigvals.npy"
            if p.is_file():
                eigs.append(np.load(p))
        if not eigs:
            ax.set_title(f"{lbl} (no data)")
            continue
        plot_spectrum(ax, np.concatenate(eigs), bins=60)
        ax.set_title(lbl)
    fig.savefig(outfile)
    plt.close(fig)


def extract_transition_widths(rows) -> list[dict]:
    """Per seed, per transition, compute peak T and FWHM for dρ/dT and dS/dT.

    For each seed, we evaluate the derivatives on that seed's own T-sweep
    (np.gradient on the per-seed ρ(T) and S(T) curves), find the peak of
    |d/dT| inside each transition window, and measure FWHM by linear
    interpolation between sample points. Returning per-seed rows lets the
    caller aggregate the seed-to-seed scatter.
    """
    by_seed: dict[int, list[tuple[float, float, float]]] = {}
    for r in rows:
        seed = int(r["seed"])
        T = float(r["temperature"])
        rho = float(r["density"])
        S = float(r["S"])
        if not (np.isnan(rho) or np.isnan(S)):
            by_seed.setdefault(seed, []).append((T, rho, S))

    results: list[dict] = []
    for seed, items in sorted(by_seed.items()):
        items.sort()
        T = np.array([it[0] for it in items])
        rho = np.array([it[1] for it in items])
        S = np.array([it[2] for it in items])
        if T.size < 3:
            continue
        d_rho = np.gradient(rho, T)
        d_S = np.gradient(S, T)
        for transition, window in TRANSITION_WINDOWS.items():
            for obs_name, d_y in [("rho", d_rho), ("S", d_S)]:
                peak_T, peak_h, fwhm = fwhm_in_window(T, d_y, window, use_abs=True)
                results.append({
                    "seed": seed,
                    "transition": transition,
                    "observable": obs_name,
                    "peak_T": peak_T,
                    "peak_height": peak_h,
                    "fwhm": fwhm,
                })
    return results


def summarise_widths(width_rows) -> list[dict]:
    """Aggregate the per-seed width table into (transition, observable) means.

    Drops NaNs (cases where the derivative didn't form a clean peak inside
    the window) before averaging.
    """
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in width_rows:
        key = (r["transition"], r["observable"])
        g = groups.setdefault(key, {"peak_T": [], "peak_height": [], "fwhm": []})
        for k in g:
            v = float(r[k])
            if not np.isnan(v):
                g[k].append(v)
    out = []
    for (transition, obs), vals in sorted(groups.items()):
        row = {"transition": transition, "observable": obs}
        for k, xs in vals.items():
            row[f"{k}_mean"] = float(np.mean(xs)) if xs else float("nan")
            row[f"{k}_std"] = float(np.std(xs)) if xs else float("nan")
            row[f"{k}_n_seeds"] = len(xs)
        out.append(row)
    return out


def width_ratio_table(summary) -> list[dict]:
    """For each transition, report FWHM(rho) / FWHM(S) — the headline number.

    > 1 means dS/dT is sharper than dρ/dT.
    """
    by_transition: dict[str, dict[str, dict]] = {}
    for s in summary:
        by_transition.setdefault(s["transition"], {})[s["observable"]] = s
    out = []
    for transition, by_obs in sorted(by_transition.items()):
        if "rho" not in by_obs or "S" not in by_obs:
            continue
        f_rho = by_obs["rho"]["fwhm_mean"]
        f_S = by_obs["S"]["fwhm_mean"]
        s_rho = by_obs["rho"]["fwhm_std"]
        s_S = by_obs["S"]["fwhm_std"]
        ratio = f_rho / f_S if f_S and not np.isnan(f_S) else float("nan")
        # Gaussian error propagation on the ratio.
        if not np.isnan(ratio) and f_rho and f_S:
            rel = np.sqrt((s_rho / f_rho) ** 2 + (s_S / f_S) ** 2)
            ratio_err = ratio * rel
        else:
            ratio_err = float("nan")
        out.append({
            "transition": transition,
            "fwhm_rho": f_rho,
            "fwhm_rho_err": s_rho,
            "fwhm_S": f_S,
            "fwhm_S_err": s_S,
            "fwhm_ratio_rho_over_S": ratio,
            "fwhm_ratio_err": ratio_err,
        })
    return out


def _write_csv(rows, outfile: Path) -> None:
    if not rows:
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(outfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    apply_style()
    rows = _load_summary(args.output_dir / "lj_summary.csv")
    figs = args.output_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_density_entropy(rows, figs / "fig_lj_density_entropy.pdf")
    fig_spectra_by_phase(rows, args.output_dir / "runs", figs / "fig_lj_spectra_by_phase.pdf")

    width_rows = extract_transition_widths(rows)
    _write_csv(width_rows, args.output_dir / "transition_widths_per_seed.csv")
    summary = summarise_widths(width_rows)
    _write_csv(summary, args.output_dir / "transition_widths_summary.csv")
    _write_csv(width_ratio_table(summary), args.output_dir / "transition_width_ratios.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
