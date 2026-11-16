#!/usr/bin/env python3
"""Figures and tables from damage_ingest.py output (one sweep tree).

    python damage_report.py \\
        --results /work/stovey/ballistic-diamond/analysis/damage-maps \\
        --out figures/final --temperature 300

Writes, for the gridded channel-map cells of one orientation / energy /
temperature:

    depth_channel_maps.png      Sn stop depth vs entry position, per tilt
    damage_channel_maps.png     vacancies within 10 nm of the stopped Sn vs
                                entry position, per tilt
    damage_vs_depth.png         local damage against how deep the ion went
    vacancy_clusters.png        isolated / divacancy / larger clusters near
                                the ion, and the share of clean impacts
    vacancy_depth_profiles.png  where the vacancies are vs where the Sn is
    damage_tables.md            the numbers behind all of it

Colour follows one rule per job: a single-hue ramp for magnitude, fixed
categorical slots for identity; text is never drawn in a series colour.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
import fold_channel_map as fcm  # noqa: E402  (projected lattice + overlay, box frame)

SURFACE, INK, INK2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dc"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.frameon": False,
})


def tilt_label(angle: float) -> str:
    return f"{angle:g}°"


def load(results: Path, temperature: float, energy: float, orientation: int) -> pd.DataFrame:
    files = sorted(results.glob("summary*.csv"))
    if not files:
        raise FileNotFoundError(f"no summary*.csv in {results}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="run", keep="last")
    df = df[(df["ok"] == 1) & (df["T_K"] == temperature) & (df["E_keV"] == energy)
            & (df["orientation"] == orientation)].copy()
    df["vac_clustered_r10"] = df["vac_r10"] - df["V1_r10"]
    df["vac_in_V2_r10"] = 2 * df["V2_r10"]
    df["vac_in_V3p_r10"] = df["vac_r10"] - df["V1_r10"] - 2 * df["V2_r10"]
    return df.sort_values(["angle_deg", "ensemble"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Channel maps
# ---------------------------------------------------------------------------

def channel_maps(df, column, cbar_label, title, out, vmax=None, half=3.0):
    lat = fcm.projected_lattice()
    cols, segs = fcm.crystal_overlay(half)
    angles = sorted(df["angle_deg"].unique())
    vmax = float(vmax if vmax is not None else np.nanpercentile(df[column], 99))

    fig, axes = plt.subplots(1, len(angles), figsize=(3.05 * len(angles) + 0.9, 3.9),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    g = np.linspace(-half, half, 360)
    gx, gy = np.meshgrid(g, g)
    from scipy.interpolate import griddata
    for k, (ax, ang) in enumerate(zip(axes, angles)):
        sub = df[(df["angle_deg"] == ang) & df["entry_x"].notna()]
        pts = fcm.fold(sub[["entry_x", "entry_y"]].to_numpy(), lat)
        val = sub[column].to_numpy(dtype=float)
        reps = int(np.ceil(2 * half / np.linalg.norm(lat[0]))) + 2
        P, V = [], []
        for i, j in itertools.product(range(-reps, reps + 1), repeat=2):
            sh = pts + i * lat[0] + j * lat[1]
            m = (np.abs(sh[:, 0]) < half + 1.2) & (np.abs(sh[:, 1]) < half + 1.2)
            if m.any():
                P.append(sh[m]); V.append(val[m])
        field = griddata(np.vstack(P), np.concatenate(V), (gx, gy), method="linear")
        im = ax.imshow(field, origin="lower", cmap=SEQ, vmin=0, vmax=vmax,
                       extent=[-half, half, -half, half], interpolation="bilinear", zorder=1)
        ax.add_collection(LineCollection(segs, colors=INK2, linewidths=0.7, alpha=0.38, zorder=2))
        m = (np.abs(cols[:, 0]) <= half) & (np.abs(cols[:, 1]) <= half)
        ax.scatter(cols[m, 0], cols[m, 1], s=15, c=SURFACE, edgecolors=INK, linewidths=0.6, zorder=3)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
        ax.set_title(f"{tilt_label(ang)} tilt", loc="left")
        ax.set_xlabel("entry x [Å]")
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        if k == 0:
            ax.set_ylabel("entry y [Å]")
        else:
            ax.set_yticklabels([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.012)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5)
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Damage against depth
# ---------------------------------------------------------------------------

def damage_vs_depth(df, out, cone=2.0):
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    inside = df["angle_deg"] <= cone
    for mask, colour, label in ((inside, S1, f"tilt ≤ {cone:g}° (inside the channeling cone)"),
                                (~inside, S2, f"tilt ≥ 5° (outside it)")):
        sub = df[mask]
        ax.scatter(sub["depth_stop_nm"], sub["vac_r10"], s=46, color=colour, alpha=0.85,
                   edgecolors=SURFACE, linewidths=1.2, label=f"{label}, n={len(sub)}", zorder=3)
    ax.set_xlabel("Sn stop depth [nm]")
    ax.set_ylabel("vacancies within 10 nm of the stopped Sn")
    ax.set_xlim(left=0); ax.set_ylim(bottom=-6)
    ax.grid(axis="y", color=RULE, linewidth=0.8, zorder=0)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Ions that channel deep stop in almost undamaged crystal", loc="left")
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vacancy clusters near the ion
# ---------------------------------------------------------------------------

def vacancy_clusters(df, out):
    angles = sorted(df["angle_deg"].unique())
    x = np.arange(len(angles))
    means = df.groupby("angle_deg")[["V1_r10", "vac_in_V2_r10", "vac_in_V3p_r10"]].mean().loc[angles]
    no_cluster = ((df["V2_r10"] + df["V3p_r10"]) == 0).astype(float)
    clean = no_cluster.groupby(df["angle_deg"]).mean().loc[angles]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    bottom = np.zeros(len(angles))
    for col, colour, label in (("V1_r10", S1, "isolated vacancies"),
                               ("vac_in_V2_r10", S2, "in divacancies"),
                               ("vac_in_V3p_r10", S3, "in clusters of 3+")):
        a1.bar(x, means[col], bottom=bottom, width=0.62, color=colour, label=label,
               edgecolor=SURFACE, linewidth=1.6, zorder=3)
        bottom += means[col].to_numpy()
    for xi, tot in zip(x, bottom):
        a1.text(xi, tot + bottom.max() * 0.015, f"{tot:.0f}", ha="center", va="bottom",
                fontsize=10, color=INK)
    a1.set_xticks(x, [tilt_label(a) for a in angles])
    a1.set_xlabel("beam tilt")
    a1.set_ylabel("vacancies within 10 nm of the Sn (mean per ion)")
    a1.set_title("How the vacancies near the ion are grouped", loc="left")
    a1.grid(axis="y", color=RULE, linewidth=0.8, zorder=0)
    a1.legend(loc="upper left", fontsize=10)

    a2.bar(x, 100 * clean.to_numpy(), width=0.62, color=S1, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    for xi, v in zip(x, 100 * clean.to_numpy()):
        a2.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color=INK)
    a2.set_xticks(x, [tilt_label(a) for a in angles])
    a2.set_ylim(0, 108); a2.set_yticks([0, 25, 50, 75, 100])
    a2.set_xlabel("beam tilt")
    a2.set_ylabel("share of impacts [%]")
    a2.set_title("Impacts with no divacancy or larger cluster within 10 nm", loc="left")
    a2.grid(axis="y", color=RULE, linewidth=0.8, zorder=0)
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vacancy depth profiles
# ---------------------------------------------------------------------------

def vacancy_depth_profiles(df, results: Path, out, bin_nm=10.0, max_nm=320.0):
    angles = sorted(df["angle_deg"].unique())
    bins = np.arange(0.0, max_nm + bin_nm, bin_nm)
    fig, axes = plt.subplots(len(angles), 1, figsize=(8.6, 1.75 * len(angles) + 0.9),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    stats = {}
    for ax, ang in zip(axes, angles):
        sub = df[df["angle_deg"] == ang]
        depths = []
        for run in sub["run"]:
            f = results / "defects" / Path(run).with_suffix(".npz")
            if not f.exists():
                continue
            with np.load(f) as z:
                depths.append((float(z["z_surface"]) - z["vac_pos"][:, 2]) / 10.0)
        vac = np.concatenate(depths) if depths else np.zeros(0)
        hv, _ = np.histogram(vac, bins=bins)
        hs, _ = np.histogram(sub["depth_stop_nm"], bins=bins)
        fv = hv / max(hv.sum(), 1)
        fs = hs / max(hs.sum(), 1)
        ax.bar(bins[:-1], fv, width=bin_nm, align="edge", color=S1, edgecolor=SURFACE,
               linewidth=0.8, label="vacancies", zorder=2)
        ax.step(np.r_[bins[:-1], bins[-1]], np.r_[fs, fs[-1]], where="post", color=S2,
                linewidth=2.0, label="stopped Sn", zorder=3)
        ax.set_title(f"{tilt_label(ang)} tilt", loc="left", fontsize=11)
        ax.grid(axis="y", color=RULE, linewidth=0.8, zorder=0)
        stats[ang] = (len(vac) / max(len(sub), 1), float(np.median(vac)) if len(vac) else float("nan"))
    axes[0].legend(loc="upper right", ncols=2, fontsize=10)
    axes[-1].set_xlabel("depth below the surface [nm]")
    axes[len(axes) // 2].set_ylabel(f"fraction per {bin_nm:g} nm")
    axes[-1].set_xlim(0, max_nm)
    fig.suptitle("Where the vacancies are, and where the Sn ends up", x=0.01, ha="left", fontsize=12.5)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def tables(df, profile_stats) -> str:
    lines = []
    def row(label, d):
        clustered = (d["V2_r10"] + d["V3p_r10"]) > 0
        return (f"| {label} | {len(d)} | {d['depth_stop_nm'].median():.0f} | {d['n_vac'].mean():.0f} | "
                f"{d['vac_r10'].mean():.0f} | {d['vac_r10'].median():.0f} | "
                f"{100 * (d['vac_r10'] == 0).mean():.0f}% | {100 * (~clustered).mean():.0f}% | "
                f"{d['V2_r10'].mean():.1f} | {d['V3p_r10'].mean():.1f} |")
    head = ("| {} | impacts | median depth [nm] | vacancies / ion (whole track) | within 10 nm of Sn (mean) | "
            "within 10 nm (median) | impacts with 0 vacancies within 10 nm | impacts with no V2+ within 10 nm | "
            "divacancies within 10 nm (mean) | clusters of 3+ within 10 nm (mean) |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    lines += ["### Damage by beam tilt", "", head.format("tilt")]
    for ang in sorted(df["angle_deg"].unique()):
        lines.append(row(tilt_label(ang), df[df["angle_deg"] == ang]))

    lines += ["", "### Damage by how deep the ion went (all tilts pooled)", "", head.format("stop depth")]
    edges = [0, 50, 100, 200, 1000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = df[(df["depth_stop_nm"] >= lo) & (df["depth_stop_nm"] < hi)]
        if len(d):
            lines.append(row(f"{lo}–{hi} nm" if hi < 1000 else f"> {lo} nm", d))

    lines += ["", "### Where the Sn sits (vacant lattice sites within 2 Å of the ion)", "",
              "| tilt | impacts | none (interstitial Sn) | one (substitutional-like) | two or more (split-vacancy-like) |",
              "|---|---|---|---|---|"]
    for ang in sorted(df["angle_deg"].unique()):
        d = df[df["angle_deg"] == ang]
        n0 = (d["sn_nvac_2A"] == 0).mean(); n1 = (d["sn_nvac_2A"] == 1).mean(); n2 = (d["sn_nvac_2A"] >= 2).mean()
        lines.append(f"| {tilt_label(ang)} | {len(d)} | {100*n0:.0f}% | {100*n1:.0f}% | {100*n2:.0f}% |")

    lines += ["", "### Analysis quality", "",
              "| quantity | median | worst |", "|---|---|---|",
              f"| atom–site residual after registration [Å] | {df['resid_median'].median():.3f} | {df['resid_median'].max():.3f} |",
              f"| largest slice-to-slice step of the displacement field [Å] | {df['reg_max_step'].median():.3f} | {df['reg_max_step'].max():.3f} |",
              f"| slab breathing amplitude removed [Å] | {df['reg_u_range'].median():.2f} | {df['reg_u_range'].max():.2f} |",
              f"| false vacancies in pristine control region (ions deeper than 50 nm) | "
              f"{df[df['depth_stop_nm'] > 50]['control_vac'].median():.0f} | {df[df['depth_stop_nm'] > 50]['control_vac'].max():.0f} |",
              f"| anchor-scaffold damage height above slab bottom [Å] | {df['scaffold_top'].median():.0f} | {df['scaffold_top'].max():.0f} |",
              f"| snippets clear of the scaffold | {int(df['snippet_clean'].sum())} of {len(df)} | |"]
    if profile_stats:
        lines += ["", "### Vacancy depth", "", "| tilt | vacancies per ion | median vacancy depth [nm] |", "|---|---|---|"]
        for ang, (per_ion, med) in profile_stats.items():
            lines.append(f"| {tilt_label(ang)} | {per_ion:.0f} | {med:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="damage_ingest.py --out directory")
    ap.add_argument("--out", type=Path, default=Path("figures/final"))
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--energy", type=float, default=35.0)
    ap.add_argument("--orientation", type=int, default=110)
    args = ap.parse_args()

    df = load(args.results, args.temperature, args.energy, args.orientation)
    if df.empty:
        print("no matching impacts in summary.csv")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    head = (f"Sn {args.energy:g} keV → diamond ⟨{args.orientation}⟩, {args.temperature:g} K")
    channel_maps(df, "depth_stop_nm", "Sn stop depth [nm]",
                 f"{head} — stop depth by where the ion enters the channel cell",
                 args.out / "depth_channel_maps.png", vmax=300.0)
    channel_maps(df, "vac_r10", "vacancies within 10 nm of the Sn",
                 f"{head} — damage left around the stopped ion, by entry position",
                 args.out / "damage_channel_maps.png")
    damage_vs_depth(df, args.out / "damage_vs_depth.png")
    vacancy_clusters(df, args.out / "vacancy_clusters.png")
    stats = vacancy_depth_profiles(df, args.results, args.out / "vacancy_depth_profiles.png")
    text = f"## {head}: damage summary\n\n" + tables(df, stats)
    (args.out / "damage_tables.md").write_text(text)
    print(text)
    print(f"figures and tables -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
