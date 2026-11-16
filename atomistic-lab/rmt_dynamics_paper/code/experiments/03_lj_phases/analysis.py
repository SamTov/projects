"""Experiment 3 post-processing: density, entropy and diffusion along the
isobar for each starting state, phase spectra, and FWHM extraction for the
`S(C)` vs `ρ` transition-resolution claim.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics import fwhm_in_window
from rmt_dynamics.plotting import apply_style, plot_spectrum

# Windows in T (K) searched for each first-order transition on the 20 atm
# isobar. They are deliberately wide: the heating branch (fcc start) melts
# above coexistence and the dense branches boil above it, while the dilute
# branch condenses below it.
TRANSITION_WINDOWS = {
    "melt": (50.0, 115.0),
    "boil": (95.0, 155.0),
}
BRANCHES = {"fcc": ("C0", "crystal start"), "dense": ("C1", "dense start"), "dilute": ("C2", "dilute start")}
# (label, starting state, temperature) for the representative spectra.
PHASE_EXAMPLES = [("solid", "fcc", 40.0), ("liquid", "dense", 100.0), ("vapor", "dilute", 150.0)]


def _load_summary(path: Path) -> list[dict]:
    with open(path) as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for r in rows:
        r.setdefault("init", "dense")
    return rows


def _aggregate(rows, key, init):
    out = {}
    for r in rows:
        if r["init"] != init or r.get(key) in (None, ""):
            continue
        v = float(r[key])
        if not np.isnan(v):
            out.setdefault(float(r["temperature"]), []).append(v)
    Ts = sorted(out)
    means = np.array([np.mean(out[T]) for T in Ts])
    stds = np.array([np.std(out[T]) for T in Ts])
    return np.array(Ts), means, stds


def fig_density_entropy(rows, outfile: Path) -> None:
    panels = [
        ("density", r"density (g/cm$^3$)"),
        ("S_over_logN", r"$S(C) / \ln N$"),
        ("diffusion", r"$\mathrm{Tr}(C)/N$ ($\mathrm{\AA}^2$/fs)"),
    ]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(5, 10), sharex=True, constrained_layout=True)
    for init, (color, label) in BRANCHES.items():
        curves = {}
        for ax, (key, ylabel) in zip(axes, panels):
            Ts, y, err = _aggregate(rows, key, init)
            if Ts.size == 0:
                continue
            ax.errorbar(Ts, y, yerr=err, marker="o", markersize=3, capsize=2, color=color, label=label)
            ax.set_ylabel(ylabel)
            curves[key] = (Ts, y)
        # Derivatives, each scaled to its own maximum so the widths compare.
        for key, style in [("density", "-"), ("S_over_logN", "--")]:
            if key in curves and curves[key][0].size >= 3:
                Ts, y = curves[key]
                d = np.abs(np.gradient(y, Ts))
                if d.max() > 0:
                    axes[-1].plot(Ts, d / d.max(), style, color=color)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("T (K)")
    axes[-1].set_ylabel(r"$|d/dT|$, scaled (solid: $\rho$, dashed: $S$)")
    fig.savefig(outfile)
    plt.close(fig)


def fig_spectra_by_phase(rows, runs_dir: Path, outfile: Path) -> None:
    """Pooled eigenvalue histogram at one representative state per phase."""
    fig, axes = plt.subplots(1, len(PHASE_EXAMPLES), figsize=(11, 3.2), constrained_layout=True)
    for ax, (phase, init, T) in zip(axes, PHASE_EXAMPLES):
        lbl = f"{phase} ({init} start, T={T:g} K)"
        eigs = []
        for r in rows:
            if r["init"] != init or float(r["temperature"]) != T:
                continue
            p = runs_dir / f"init={init}_temperature={T:g}_seed={int(r['seed'])}" / "eigvals.npy"
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
    """Per starting state and seed, peak T and FWHM of dρ/dT and dS/dT.

    The derivatives are taken on each run series' own T-sweep; the peak of
    |d/dT| is located inside each transition window and its FWHM measured by
    linear interpolation between sample points. Per-seed rows let the caller
    aggregate the seed-to-seed scatter.
    """
    series: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for r in rows:
        T = float(r["temperature"])
        rho = float(r["density"])
        S = float(r["S"])
        if not (np.isnan(rho) or np.isnan(S)):
            series.setdefault((r["init"], int(r["seed"])), []).append((T, rho, S))

    results: list[dict] = []
    for (init, seed), items in sorted(series.items()):
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
                    "init": init,
                    "seed": seed,
                    "transition": f"{transition}/{init}",
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
