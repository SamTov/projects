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
def wishart_entropy_table(N: int, qs: np.ndarray, reps: int = 16, seed: int = 1) -> np.ndarray:
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
        ax.set_xlabel(r"eigenvalue $\mu / s^2_{\mathrm{free}}$")
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
    ax.set_xlabel(r"eigenvalue $\mu/s^2$")
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


# ----------------------------------------------------------- molten salt ---
SALT_BRANCHES = [("crystal", "crystal start"), ("liquid", "liquid start")]


def _branch_stats(rows, key, init, f=lambda r, v: v):
    groups = defaultdict(list)
    for r in rows:
        if r["init"] == init:
            groups[float(r["temperature"])].append(f(r, fnum(r[key])))
    Ts = np.array(sorted(groups))
    return Ts, np.array([np.mean(groups[T]) for T in Ts]), np.array([np.std(groups[T]) for T in Ts])


def salt_figure(res: Path, outfile: Path, N: int = 512) -> list[dict]:
    rows = read_csv(res / "05_molten_salt" / "salt_summary.csv")
    qs = np.geomspace(0.01, 0.6, 14)
    tab = wishart_entropy_table(N, qs)
    null = lambda q: np.interp(np.log(q), np.log(qs), tab)
    D = lambda r, v: v  # identity
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH * 0.8, 4.4), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.1, hspace=0.1)
    ax_rho, ax_sig, ax_del, ax_ent = axes.flat
    for (init, lbl), color, mk in zip(SALT_BRANCHES, SERIES, MARKERS):
        T, y, e = _branch_stats(rows, "density", init)
        ax_rho.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
        ex = lambda r, v: null(N / fnum(r["T_eff"])) - v
        T, y, e = _branch_stats(rows, "S_over_logN", init, ex)
        ax_ent.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
    # Transport only where the ions diffuse (melted states on either branch).
    melted = [r for r in rows if fnum(r["density"]) < 1.6]
    groups = defaultdict(list)
    for r in melted:
        groups[float(r["temperature"])].append(r)
    T = np.array(sorted(groups))
    def col(key):
        v = [[fnum(r[key]) for r in groups[t]] for t in T]
        return np.array([np.mean(x) for x in v]), np.array([np.std(x) for x in v])
    sig, sig_e = col("sigma_S_per_m")
    ne, ne_e = col("sigma_NE_S_per_m")
    ax_sig.errorbar(T, ne, yerr=ne_e, color=MUTED, marker="s", mfc="white", label="Nernst–Einstein")
    ax_sig.errorbar(T, sig, yerr=sig_e, color=SERIES[1], marker="o", label="Green–Kubo")
    dm, dm_e = col("Delta_sigma_matrix")
    dt, dt_e = col("Delta_sigma_from_time_GK")
    ax_del.errorbar(T, dm, yerr=dm_e, color=SERIES[1], marker="o", label="block sums of $C$")
    ax_del.errorbar(T + 12, dt, yerr=dt_e, color=INK_2, marker="s", mfc="white", ls="--",
                    label="charge-current GK")
    ax_del.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax_rho.set_ylabel(r"density (g cm$^{-3}$)")
    ax_sig.set_ylabel(r"$\sigma$ (S m$^{-1}$)")
    ax_del.set_ylabel(r"$\Delta^{\sigma}$")
    ax_ent.set_ylabel(r"$\Delta S/\ln|\mathcal{N}|$")
    ax_ent.axhline(0, color=MUTED, lw=0.8, ls="--")
    for ax in axes.flat:
        ax.set_xlabel("T (K)")
        ax.set_xlim(850, 1850)
    soft_legend(ax_rho, loc="upper right")
    soft_legend(ax_sig, loc="center right")
    soft_legend(ax_del, loc="lower left")
    soft_legend(ax_ent, loc="center right")
    for ax, l in zip(axes.flat, "abcd"):
        panel_label(ax, l)
    fig.savefig(outfile)
    plt.close(fig)
    return [{"T": t, "sigma": a, "sigma_NE": b, "Delta_matrix": c, "Delta_GK": d}
            for t, a, b, c, d in zip(T, sig, ne, dm, dt)]


# ----------------------------------------------------------------- water ---
WATER_BRANCHES = [("ice", "ice start"), ("liquid", "liquid start")]


def water_figure(res: Path, outfile: Path) -> None:
    rows = read_csv(res / "04_water_phases" / "water_summary.csv")
    ov_rows = read_csv(res / "04_water_phases" / "intramolecular_overlap.csv")
    NO = 432
    qs = np.geomspace(0.01, 0.8, 16)
    tab = wishart_entropy_table(NO, qs)
    null = lambda q: np.interp(np.log(q), np.log(qs), tab)
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08, hspace=0.1)
    a_rho, a_D, a_S, a_spec, a_ov, a_hb = axes.flat
    for (init, lbl), color, mk in zip(WATER_BRANCHES, SERIES, MARKERS):
        T, y, e = _branch_stats(rows, "density", init)
        a_rho.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
        T, y, e = _branch_stats(rows, "diffusion_O", init, lambda r, v: v * 1e4)
        a_D.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
        T, y, e = _branch_stats(rows, "S_oxygens_over_logN", init,
                                lambda r, v: null(NO / fnum(r["T_eff_oxygens"])) - v)
        a_S.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
        T, y, e = _branch_stats(ov_rows, "overlap_intramolecular", init)
        a_ov.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
        T, y, e = _branch_stats(rows, "hbond_spearman_top5", init)
        a_hb.errorbar(T, y, yerr=e, color=color, marker=mk, label=lbl)
    # All-atom spectrum of the liquid at 300 K: two intramolecular modes per
    # molecule sit far below the molecular bulk.
    eigs = [np.load(p) for p in sorted((res / "04_water_phases" / "runs").glob(
        "init=liquid_temperature=300_seed=*/eigvals_all.npy"))]
    mu = np.concatenate([e / e[2 * NO:].mean() for e in eigs])
    bins = np.geomspace(1e-3, 5, 70)
    a_spec.hist(mu, bins=bins, histtype="stepfilled", color=SERIES[1], alpha=0.35, lw=0)
    a_spec.hist(mu, bins=bins, histtype="step", color=SERIES[1], lw=1.2)
    cut = np.sort(eigs[0])[2 * NO - 1] / eigs[0][2 * NO:].mean()
    a_spec.axvline(cut, color=MUTED, ls="--", lw=1.0)
    top = a_spec.get_ylim()[1] * 1.22
    a_spec.set_ylim(0, top)
    a_spec.text(cut * 0.85, top * 0.97, f"{2 * NO} lowest", ha="right", va="top", fontsize=7, color=INK_2)
    a_spec.text(cut * 1.18, top * 0.97, f"{NO} highest", ha="left", va="top", fontsize=7, color=INK_2)
    a_spec.set_xscale("log")
    a_spec.set_xlabel(r"eigenvalue $\mu/\langle\mu\rangle_{\mathrm{mol}}$")
    a_spec.set_ylabel("count")
    a_spec.set_title("all atoms, liquid 300 K", color=INK_2, loc="left")
    a_ov.axhline(2 / 3, color=MUTED, ls=":", lw=1.0)
    a_ov.text(205, 2 / 3 + 0.01, "random modes", fontsize=7, color=INK_2, va="bottom")
    a_ov.set_ylim(0.6, 1.02)
    a_hb.axhline(0, color=MUTED, ls="--", lw=0.8)
    a_rho.set_ylabel(r"density (g cm$^{-3}$)")
    a_D.set_ylabel(r"$\mathrm{Tr}\,C_{\mathrm{O}}/|\mathcal{N}_{\mathrm{O}}|$ ($10^{-4}$ Å$^2$ fs$^{-1}$)")
    a_D.set_yscale("log")
    a_S.set_ylabel(r"$\Delta S_{\mathrm{O}}/\ln|\mathcal{N}_{\mathrm{O}}|$")
    a_S.axhline(0, color=MUTED, lw=0.8, ls="--")
    a_ov.set_ylabel("overlap with intramolecular space")
    a_hb.set_ylabel("Spearman $\\rho$, top modes vs H-bonds")
    for ax in (a_rho, a_D, a_S, a_ov, a_hb):
        ax.set_xlabel("T (K)")
        ax.set_xlim(192, 348)
    soft_legend(a_rho, loc="center right", bbox_to_anchor=(1.0, 0.45))
    soft_legend(a_hb, loc="lower left")
    for ax, l in zip(axes.flat, "abcdef"):
        panel_label(ax, l)
    fig.savefig(outfile)
    plt.close(fig)


# -------------------------------------------------------------- concept ---
def concept_figure(res: Path, figs: Path, outfile: Path, k: int = 27, seed: int = 1) -> None:
    """Render of a bonded liquid, its correlation matrix, and its spectrum."""
    from matplotlib.colors import LinearSegmentedColormap
    from rmt_dynamics import load_C

    name = "02_bond_addition_long"
    rows, counts, s2, T_eff, _ = _bond_tables(res, name)
    run = res / name / "runs" / f"k_bonds={k}_seed={seed}"
    C, _ = load_C(run / "C.npz")
    bonds = np.load(run / "bonds.npy")
    N = C.shape[0]
    # Show isolated bonds (atoms in exactly one bond) so each pair is a clean
    # 2 x 2 block, followed by an equal number of free atoms.
    uses = np.bincount(bonds.ravel(), minlength=N)
    iso = [(int(i), int(j)) for i, j in bonds if uses[i] == 1 and uses[j] == 1][:8]
    in_bond = set(int(x) for x in bonds.ravel())
    free = [i for i in range(N) if i not in in_bond]
    d_free = float(np.mean(np.diag(C)[free]))
    bonded = [a for pair in iso for a in pair]
    order = bonded + free[: len(bonded)]
    sub = C[np.ix_(order, order)] / d_free

    fig = plt.figure(figsize=(FULL_WIDTH, 2.55), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 1.12])
    ax_r, ax_c, ax_s = (fig.add_subplot(gs[0, i]) for i in range(3))

    img = plt.imread(figs / "render_bonded_liquid.png")
    h, w = img.shape[:2]
    m = int(0.06 * h)
    ax_r.imshow(img[m:h - m, m:w - m])
    ax_r.set_axis_off()
    ax_r.set_title(f"{k} bonds, $|\\mathcal{{N}}| = {N}$", color=INK_2, loc="left")

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    im = ax_c.imshow(sub, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax_c.grid(False)
    nb = len(bonded)
    for x in (nb - 0.5,):
        ax_c.axhline(x, color=INK_2, lw=0.6)
        ax_c.axvline(x, color=INK_2, lw=0.6)
    ax_c.set_xticks([nb / 2, nb + (len(order) - nb) / 2])
    ax_c.set_xticklabels(["bonded", "free"])
    ax_c.set_yticks([nb / 2, nb + (len(order) - nb) / 2])
    ax_c.set_yticklabels(["bonded", "free"], rotation=90, va="center")
    ax_c.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax_c, shrink=0.85, pad=0.02)
    cb.set_label(r"$C_{ij}/C^{\mathrm{free}}_{ii}$")
    cb.outline.set_visible(False)
    ax_c.set_title(f"$C$, {len(iso)} bonded pairs and {len(bonded)} free atoms", color=INK_2, loc="left")

    mu = np.sort(np.linalg.eigvalsh(C)) / s2
    mu0 = np.sort(np.concatenate([np.load(p) for p in sorted((res / name / "runs").glob(
        "k_bonds=0_seed=*/eigvals.npy"))])) / s2
    lo, hi = mp_edges(N, T_eff, 1.0)
    ax_s.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax_s.text(N * 0.98, (lo * hi) ** 0.5, "noise bulk", ha="right", va="center", fontsize=7, color=INK_2)
    ranks = np.arange(1, N + 1)
    ax_s.plot(np.linspace(1, N, mu0.size), mu0, color=MUTED, lw=1.0, label="no bonds")
    ax_s.plot(ranks[k + 1:], mu[k + 1:], color=INK_2, lw=1.4, label=f"{k} bonds")
    ax_s.plot(ranks[1:k + 1], mu[1:k + 1], "o", color=SERIES[0], ms=3.2, label=f"{k} bond modes")
    ax_s.plot(ranks[:1], np.maximum(mu[:1], 2e-3), "D", color=INK_2, ms=3.2, mfc="white",
              label="momentum mode")
    ax_s.set_yscale("log")
    ax_s.set_ylim(2e-3, 4)
    ax_s.set_xlim(-4, N + 4)
    ax_s.set_xlabel("eigenvalue rank")
    ax_s.set_ylabel(r"$\mu/s^{2}_{\mathrm{free}}$")
    ax_s.set_title(r"spectrum, $\tau_{\max} = 100$ ps", color=INK_2, loc="left")
    soft_legend(ax_s, loc="lower right")
    for ax, l in zip((ax_r, ax_c, ax_s), "abc"):
        panel_label(ax, l)
    fig.savefig(outfile)
    plt.close(fig)


def main() -> int:
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    figs.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    noise_figure(res, figs / "fig_noise_null.pdf")
    if (figs / "render_bonded_liquid.png").is_file():
        concept_figure(res, figs, figs / "fig_concept.pdf")
    bonding_figure(res, figs / "fig_bonding.pdf")
    table = lj_figure(res, figs / "fig_lj_phases.pdf")
    if (res / "05_molten_salt" / "salt_summary.csv").is_file():
        salt_figure(res, figs / "fig_salt.pdf")
    if (res / "04_water_phases" / "intramolecular_overlap.csv").is_file():
        water_figure(res, figs / "fig_water.pdf")
    with open(res / "03_lj_phases" / "entropy_excess.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
