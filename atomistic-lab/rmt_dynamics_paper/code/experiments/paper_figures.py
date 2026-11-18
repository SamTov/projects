"""Manuscript figures, built as multi-panel blocks from the copied result tables.

Usage: python paper_figures.py <results_dir> <figures_dir>

Reads, under <results_dir>:
  02_bond_addition/{bond_summary.csv, bond_mode_counts.csv, runs/*/eigvals.npy,
                    lag_scan_k16_seed1.txt}
  02_bond_addition_long/{bond_summary.csv, bond_mode_counts.csv, runs/*/eigvals.npy}
  03_lj_phases/{lj_summary.csv, runs/*/eigvals.npy}
and writes fig_bonding.pdf and fig_lj_phases.pdf (plus entropy_excess.csv).

Colour follows a validated categorical order (blue, orange, aqua: all
pairs pass the CVD and normal-vision checks) and a single-hue blue ramp
for ordered magnitudes; every coloured series also has its own marker or
line style and a legend, so identity never rests on colour alone.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from rmt_dynamics import mp_density, mp_edges

# ---------------------------------------------------------------- style ---
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
MARKERS = ["o", "s", "^"]

FULL_WIDTH = 7.0   # inches; figures are placed at \linewidth


def apply_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "errorbar.capsize": 0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def panel_label(ax, letter: str) -> None:
    """Bold panel letter outside the axes, clear of tick labels."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-26, 4),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10, fontweight="bold", color=INK)


def soft_legend(ax, **kw):
    leg = ax.legend(**kw)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return leg


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------- null model ---
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


def mp_curve(ax, N, T_eff, x_max, color=INK_2, label="Marchenko–Pastur"):
    """MP density for unit variance, plotted against mu / sigma^2."""
    lo, hi = mp_edges(N, T_eff, 1.0)
    x = np.linspace(lo, hi, 400)
    ax.plot(x, mp_density(x, N, T_eff, 1.0), color=color, lw=1.0, ls="--", label=label)
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    return lo, hi


# ------------------------------------------------------------- bonding ---
def _bond_tables(res: Path, name: str):
    rows = read_csv(res / name / "bond_summary.csv")
    counts = {(c["k_bonds"], c["seed"]): c for c in read_csv(res / name / "bond_mode_counts.csv")}
    base = [r for r in rows if r["k_bonds"] == "0"]
    sigma2 = float(np.mean([fnum(r["sigma2"]) for r in base]))
    T_eff = float(np.mean([fnum(r["T_eff"]) for r in base]))
    S0 = float(np.mean([fnum(r["S"]) for r in base]))
    return rows, counts, sigma2, T_eff, S0


def _pooled_eigs(res: Path, name: str, k: int, sigma2: float) -> np.ndarray:
    eigs = [np.load(p) for p in sorted((res / name / "runs").glob(f"k_bonds={k}_seed=*/eigvals.npy"))]
    return np.concatenate(eigs) / sigma2 if eigs else np.array([])


def _agg_by_k(rows, counts, key, from_counts=False, offset=0.0):
    ks = sorted({int(r["k_bonds"]) for r in rows})
    xs, ys, es = [], [], []
    for k in ks:
        rs = [r for r in rows if int(r["k_bonds"]) == k]
        src = [counts[(r["k_bonds"], r["seed"])] for r in rs] if from_counts else rs
        v = np.array([fnum(s[key]) for s in src]) - offset
        nc = np.mean([fnum(r["n_constraints"]) for r in rs])
        v = v[~np.isnan(v)]
        if v.size:
            xs.append(nc); ys.append(v.mean()); es.append(v.std())
    return np.array(xs), np.array(ys), np.array(es)


def _parse_lag_scan(path: Path):
    lags, bonded, free, free_sd, edge, q = [], [], [], [], [], []
    pat = re.compile(r"t_max=\s*([\d.]+) ps\s+q=\s*([\d.]+).*bonded anti/D=([\d.]+).*unbonded anti/D=([\d.]+) \(sd ([\d.]+)\)\s+lower edge/D=([\d.]+)")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            lags.append(float(m[1])); q.append(float(m[2])); bonded.append(float(m[3]))
            free.append(float(m[4])); free_sd.append(float(m[5])); edge.append(float(m[6]))
    return [np.array(a) for a in (lags, q, bonded, free, free_sd, edge)]


def bonding_figure(res: Path, outfile: Path) -> None:
    short = _bond_tables(res, "02_bond_addition")
    long_ = _bond_tables(res, "02_bond_addition_long")
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08, hspace=0.1)

    # (a, b) spectra in units of the bond-free bulk variance.
    for ax, (rows, counts, s2, T_eff, _), name, ks, title in [
        (axes[0, 0], short, "02_bond_addition", [0, 64, 256], r"$\tau_{\max}=2$ ps, $|\mathcal{N}|=512$"),
        (axes[0, 1], long_, "02_bond_addition_long", [0, 27, 108], r"$\tau_{\max}=100$ ps, $|\mathcal{N}|=216$"),
    ]:
        N = int(round(np.load(next((res / name / "runs").glob("k_bonds=0_seed=*/eigvals.npy"))).size))
        lo, hi = mp_curve(ax, N, T_eff, 2.0)
        bins = np.linspace(0, 2.2, 56)
        for i, k in enumerate(ks):
            e = _pooled_eigs(res, name, k, s2)
            ax.hist(e, bins=bins, density=True, histtype="step", lw=1.4,
                    color=BLUE_RAMP[[0, 2, 4][i]])
            ax.plot([], [], color=BLUE_RAMP[[0, 2, 4][i]], lw=1.4, label=f"{k} bonds")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 5)
        ax.set_xlim(0, 2.2)
        ax.set_xlabel(r"eigenvalue $\mu / \sigma^2_{\mathrm{free}}$")
        ax.set_ylabel("density")
        ax.set_title(title, color=INK_2, loc="left")
        leg = soft_legend(ax, loc="lower center", bbox_to_anchor=(0.56, 0.0), ncol=2,
                          columnspacing=0.9, frameon=True, framealpha=0.92, edgecolor="none",
                          facecolor="white", borderpad=0.3)

    # (c) bonded-pair relative mode against the integration lag.
    ax = axes[0, 2]
    lags, q, bonded, free, free_sd, edge = _parse_lag_scan(res / "02_bond_addition" / "lag_scan_k16_seed1.txt")
    # Colour marks the system (orange: 2 ps set, |N| = 512; blue: 100 ps
    # set, |N| = 216), as in panels d-f; unbonded pairs are the grey reference.
    ax.errorbar(lags, free, yerr=free_sd, color=INK_2, marker="s", mfc="white", label="unbonded")
    ax.plot(lags, bonded, color=SERIES[1], marker="s", label=r"bonded ($|\mathcal{N}|$ 512)")
    ok = q < 1.0
    ax.plot(lags[ok], edge[ok], color=MUTED, ls="--", lw=1.0, marker="none", label="bulk edge")
    lrows = long_[0]
    lb = np.array([fnum(r["bond_anti_over_free"]) for r in lrows if r["k_bonds"] not in ("0",)])
    ax.errorbar([100.0], [np.nanmean(lb)], yerr=[np.nanstd(lb)], color=SERIES[0], marker="o",
                ls="none", label=r"bonded ($|\mathcal{N}|$ 216)")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"integration lag $\tau_{\max}$ (ps)")
    ax.set_ylabel(r"relative mode $/\,C_{ii}^{\mathrm{free}}$")
    soft_legend(ax, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.4)

    # (d) sub-bulk modes per constraint, (e) overlap with bond space.
    for (rows, counts, *_), color, mk, lbl in [(short, SERIES[1], "s", "2 ps"), (long_, SERIES[0], "o", "100 ps")]:
        x, y, e = _agg_by_k(rows, counts, "n_below", from_counts=True, offset=1.0)
        axes[1, 0].errorbar(x, y, yerr=e, color=color, marker=mk, label=lbl)
        x, y, e = _agg_by_k(rows, counts, "overlap_lowest")
        keep = x > 0
        axes[1, 1].errorbar(x[keep], y[keep], yerr=e[keep], color=color, marker=mk, label=lbl)
    ax = axes[1, 0]
    lim = 520
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1.0, label="one per constraint")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_xlabel("independent bond constraints")
    ax.set_ylabel("eigenvalues below bulk")
    soft_legend(ax, loc="upper left")
    ax = axes[1, 1]
    xs = np.geomspace(5, 520, 50)
    for N, ls, lbl in [(216, ":", r"chance, $|\mathcal{N}|=216$"), (512, "-.", r"chance, $|\mathcal{N}|=512$")]:
        ax.plot(xs[xs < N], xs[xs < N] / N, color=MUTED, ls=ls, lw=1.0, label=lbl)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("independent bond constraints")
    ax.set_ylabel("overlap with bond space")
    soft_legend(ax, loc="center left", bbox_to_anchor=(0.0, 0.6))

    # (f) entropy drop relative to the bond-free liquid.
    ax = axes[1, 2]
    for (rows, counts, s2, T_eff, S0), color, mk, lbl, N in [
            (short, SERIES[1], "s", "2 ps", 512), (long_, SERIES[0], "o", "100 ps", 216)]:
        ks = sorted({int(r["k_bonds"]) for r in rows})
        frac = []
        for k in ks:
            v = np.array([(S0 - fnum(r["S"])) / np.log(N) for r in rows if int(r["k_bonds"]) == k])
            frac.append((k / N, v.mean(), v.std()))
        frac = np.array(frac)
        ax.errorbar(frac[:, 0], frac[:, 1], yerr=frac[:, 2], color=color, marker=mk, label=lbl)
    ax.set_xlabel(r"bonds per atom $k/|\mathcal{N}|$")
    ax.set_ylabel(r"$[S_0 - S(C)]/\ln|\mathcal{N}|$")
    soft_legend(ax, loc="upper left")

    for ax, l in zip(axes.flat, "abcdef"):
        panel_label(ax, l)
    fig.savefig(outfile)
    plt.close(fig)


# ---------------------------------------------------------- noise null ---
def noise_figure(res: Path, outfile: Path) -> None:
    d = np.load(res / "01_noise_convergence" / "synthetic_spectrum_L50.npz")
    rows = read_csv(res / "01_noise_convergence" / "synthetic_validation.csv")
    N, s2 = int(d["N"]), float(d["sigma2"])
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.72, 2.3), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.1)
    ax = axes[0]
    mu = d["mu"] / s2
    ax.hist(mu, bins=np.linspace(0, 2.6, 52), density=True, histtype="stepfilled", color=BAND,
            edgecolor=INK_2, lw=1.0, label="spectrum of $C$")
    for T, color, ls, lbl in [(float(d["T_window"]), SERIES[0], "-", "MP, lag-window $T_{\\mathrm{eff}}$"),
                              (float(d["T_naive"]), SERIES[1], "--", "MP, $T_{\\mathrm{traj}}/\\tau_{\\mathrm{int}}$")]:
        lo, hi = mp_edges(N, T, 1.0)
        x = np.linspace(lo, hi, 400)
        ax.plot(x, mp_density(x, N, T, 1.0), color=color, ls=ls, lw=1.5, label=lbl)
    ax.set_xlabel(r"eigenvalue $\mu/\sigma^2$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 2.6)
    soft_legend(ax, loc="upper right")
    ax = axes[1]
    for kind, color, mk, lbl in [("window", SERIES[0], "o", "lag-window $T_{\\mathrm{eff}}$"),
                                 ("naive", SERIES[1], "s", "$T_{\\mathrm{traj}}/\\tau_{\\mathrm{int}}$")]:
        r = [x for x in rows if x["T_eff_kind"] == kind]
        ax.plot([fnum(x["lag_over_tau"]) for x in r], [fnum(x["KS"]) for x in r], color=color, marker=mk, label=lbl)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.set_ylim(0, 0.32)
    ax.set_xlabel(r"lag $\tau_{\max}/\tau_{\mathrm{int}}$")
    ax.set_ylabel("Kolmogorov–Smirnov distance")
    soft_legend(ax, loc="upper left")
    for ax, l in zip(axes, "ab"):
        panel_label(ax, l)
    fig.savefig(outfile)
    plt.close(fig)


# ------------------------------------------------------------ LJ phases ---
BRANCHES = [("fcc", "crystal start"), ("dense", "dense start"), ("dilute", "dilute start")]


def lj_figure(res: Path, outfile: Path, N: int = 500) -> list[dict]:
    rows = read_csv(res / "03_lj_phases" / "lj_summary.csv")
    qs = np.geomspace(0.07, 1.6, 18)
    tab = wishart_entropy_table(N, qs)
    null = lambda q: np.interp(np.log(q), np.log(qs), tab)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["init"], float(r["temperature"]))].append(r)

    fig = plt.figure(figsize=(FULL_WIDTH, 4.9), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08, hspace=0.12)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]
    table = []
    for (b, lbl), color, mk in zip(BRANCHES, SERIES, MARKERS):
        Ts = sorted(T for (bb, T) in groups if bb == b)
        cols = defaultdict(list)
        for T in Ts:
            rs = groups[(b, T)]
            f = lambda k: np.array([fnum(r[k]) for r in rs])
            qq = N / f("T_eff")
            ex = np.array([null(x) for x in qq]) - f("S_over_logN")
            for key, v in [("rho", f("density")), ("ex", ex), ("D", f("diffusion") * 1e4)]:
                cols[key].append((v.mean(), v.std()))
            cols["q"].append(qq.mean())
            table.append({"init": b, "T": T, "rho": cols["rho"][-1][0], "excess": ex.mean(),
                          "excess_sd": ex.std(), "D": cols["D"][-1][0], "q": qq.mean(),
                          "S": f("S_over_logN").mean()})
        Ts = np.array(Ts)
        q = np.array(cols["q"])
        for ax, key, mask in [(top[0], "rho", np.ones_like(Ts, bool)),
                              (top[1], "ex", q < 0.6),
                              (top[2], "D", np.ones_like(Ts, bool))]:
            m = np.array(cols[key])
            ax.errorbar(Ts[mask], m[mask, 0], yerr=m[mask, 1], color=color, marker=mk, ms=3.5,
                        lw=1.3, label=lbl)
    top[0].set_ylabel(r"density (g cm$^{-3}$)")
    top[1].set_ylabel(r"$\Delta S/\ln|\mathcal{N}|$")
    top[1].axhline(0, color=MUTED, lw=0.8, ls="--")
    top[2].set_ylabel(r"$\mathrm{Tr}\,C/|\mathcal{N}|$ ($10^{-4}$ Å$^2$ fs$^{-1}$)")
    top[2].set_yscale("log")
    for ax in top:
        ax.set_xlabel("T (K)")
        ax.set_xlim(12, 168)
        ax.set_xticks([20, 60, 100, 140])
    soft_legend(top[0], loc="lower left", bbox_to_anchor=(0.0, 0.08))
    # Phase annotations on the density panel, in ink, not series colour.
    for x, y, t in [(38, 1.82, "solid"), (78, 1.12, "liquid"), (150, 0.15, "vapour")]:
        top[0].text(x, y, t, ha="center", va="bottom", fontsize=7, color=INK_2, style="italic")

    # Bottom row: spectra in units of the mean eigenvalue, against MP at the run's q.
    examples = [("fcc", 40.0, "crystal, 40 K"), ("dense", 40.0, "disordered solid, 40 K"),
                ("dense", 100.0, "liquid, 100 K")]
    bins = np.linspace(0, 3.0, 60)
    for ax, (b, T, title), color in zip(bottom, examples, [SERIES[0], SERIES[1], SERIES[1]]):
        eigs = []
        T_effs = []
        for r in groups[(b, T)]:
            p = res / "03_lj_phases" / "runs" / f"init={b}_temperature={T:g}_seed={int(r['seed'])}" / "eigvals.npy"
            if p.is_file():
                mu = np.load(p)
                eigs.append(mu / mu.mean())
                T_effs.append(fnum(r["T_eff"]))
        mp_curve(ax, N, float(np.mean(T_effs)), 3.0)
        ax.hist(np.concatenate(eigs), bins=bins, density=True, histtype="stepfilled",
                color=color, alpha=0.35, lw=0)
        ax.hist(np.concatenate(eigs), bins=bins, density=True, histtype="step", color=color, lw=1.2,
                label="spectrum of $C$")
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 5)
        ax.set_xlim(0, 3.0)
        ax.set_title(title, color=INK_2, loc="left")
        ax.set_xlabel(r"eigenvalue $\mu/\langle\mu\rangle$")
        ax.set_ylabel("density")
    soft_legend(bottom[0], loc="upper right")
    for ax, l in zip(top + bottom, "abcdef"):
        panel_label(ax, l)
    fig.savefig(outfile)
    plt.close(fig)
    return table


def main() -> int:
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    figs.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    noise_figure(res, figs / "fig_noise_null.pdf")
    bonding_figure(res, figs / "fig_bonding.pdf")
    table = lj_figure(res, figs / "fig_lj_phases.pdf")
    with open(res / "03_lj_phases" / "entropy_excess.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
