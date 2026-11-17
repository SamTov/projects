#!/usr/bin/env python3
"""Two supplementary checks on the damage results, from the random-entry cells.

    python damage_checks.py \\
        --maps  /work/stovey/ballistic-diamond/analysis/damage-maps \\
        --sweep /work/stovey/ballistic-diamond/analysis/damage-sweep \\
        --out figures/final

1. Quench length.  Ensembles 0-14 of the random-entry cells were quenched for
   200 ps, everything else for 50 ps.  Retained vacancies depend strongly on
   how deep the ion went, so each run is compared with the 50 ps relation
   between vacancy count and stop depth measured on the gridded map campaign.

2. Lattice vibration.  The same 0.5 degree cell exists at 300 K and at
   1100 K.  A classical 300 K lattice vibrates less than real diamond (no
   zero-point motion) and 1100 K vibrates more, so the pair brackets the real
   crystal.  The vibration amplitude of each lattice is read off the
   registration residual: the distance between an atom and its site in a
   second, independent thermal snapshot has median 2.1753 * sigma_1D.

Writes temperature_damage.png and supplementary_checks.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import damage_report as rep  # noqa: E402  (shared style and palette)

RESID_TO_SIGMA = 2.1753        # median |r1 - r2| of two 3D Gaussians = 2.1753 sigma_1D
DEPTH_EDGES = np.array([0, 15, 30, 50, 75, 100, 150, 200, 250, 320])


def read(results: Path) -> pd.DataFrame:
    files = sorted(results.glob("summary*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["ok"] == 1].drop_duplicates("run", keep="last").copy()


def offset_from_map_relation(maps: pd.DataFrame, runs: pd.DataFrame, column: str = "n_vac") -> np.ndarray:
    """Each run's `column` minus the median of the 300 K, 50 ps map campaign
    at the same stop depth (bin medians, interpolated)."""
    ref = maps[maps["T_K"] == 300]
    med = ref.groupby(pd.cut(ref["depth_stop_nm"], DEPTH_EDGES), observed=False)[column].median().to_numpy()
    centres = 0.5 * (DEPTH_EDGES[:-1] + DEPTH_EDGES[1:])
    ok = ~np.isnan(med)
    return runs[column].to_numpy() - np.interp(runs["depth_stop_nm"].to_numpy(), centres[ok], med[ok])


def quench_check(maps: pd.DataFrame, sweep: pd.DataFrame) -> tuple[str, dict]:
    ref = maps[maps["T_K"] == 300]
    s = sweep[sweep["T_K"] == 300]
    resid = offset_from_map_relation(maps, s)
    long_q = resid[(s["ensemble"] < 15).to_numpy()]
    short_q = resid[(s["ensemble"] >= 15).to_numpy()]
    rng = np.random.default_rng(0)
    boot = [rng.choice(long_q, len(long_q)).mean() - rng.choice(short_q, len(short_q)).mean()
            for _ in range(20000)]
    out = dict(n_long=len(long_q), n_short=len(short_q), diff=long_q.mean() - short_q.mean(),
               lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
               scale=ref[ref["depth_stop_nm"] < 50]["n_vac"].median())
    text = (f"| quench | runs | mean offset from the 50 ps map relation [vacancies] |\n|---|---|---|\n"
            f"| 200 ps | {len(long_q)} | {long_q.mean():+.1f} |\n| 50 ps | {len(short_q)} | {short_q.mean():+.1f} |\n\n"
            f"Difference 200 ps − 50 ps: {out['diff']:+.1f} vacancies per ion "
            f"(95% interval {out['lo']:+.1f} to {out['hi']:+.1f}), against {out['scale']:.0f} "
            f"retained by a typical non-channeled ion.\n")
    return text, out


def temperature_table(a: pd.DataFrame, maps: pd.DataFrame) -> str:
    lines = ["| lattice | runs | 1D vibration amplitude [Å] | median depth [nm] | deeper than 100 nm | "
             "vacancies per ion | within 10 nm of Sn (median) | no vacancy within 10 nm | "
             "no divacancy or larger within 10 nm |", "|---|---|---|---|---|---|---|---|---|"]
    for T, d in a.groupby("T_K"):
        clean = ((d["V2_r10"] + d["V3p_r10"]) == 0).mean()
        lines.append(f"| {T:g} K | {len(d)} | {d['resid_median'].median() / RESID_TO_SIGMA:.4f} | "
                     f"{d['depth_stop_nm'].median():.0f} | {100 * (d['depth_stop_nm'] > 100).mean():.0f}% | "
                     f"{d['n_vac'].mean():.0f} | {d['vac_r10'].median():.0f} | "
                     f"{100 * (d['vac_r10'] == 0).mean():.0f}% | {100 * clean:.0f}% |")
    lines += ["", "Ions that stopped deeper than 150 nm:", "",
              "| lattice | ions | vacancies along the track (median) | within 10 nm of Sn (median) | "
              "no divacancy or larger within 10 nm | Sn with no vacant site within 2 Å |",
              "|---|---|---|---|---|---|"]
    for T, d in a[a["depth_stop_nm"] > 150].groupby("T_K"):
        clean = ((d["V2_r10"] + d["V3p_r10"]) == 0).mean()
        lines.append(f"| {T:g} K | {len(d)} | {d['n_vac'].median():.0f} | {d['vac_r10'].median():.1f} | "
                     f"{100 * clean:.0f}% | {100 * (d['sn_nvac_2A'] == 0).mean():.0f}% |")
    lines += ["", "Damage at a given depth, as the mean offset from the 300 K map relation "
              "(vacancies per ion at the same stop depth):", "",
              "| lattice | along the whole track | within 10 nm of the Sn |", "|---|---|---|"]
    rng = np.random.default_rng(1)
    for T, d in a.groupby("T_K"):
        cells = []
        for col in ("n_vac", "vac_r10"):
            off = offset_from_map_relation(maps, d, col)
            boot = [rng.choice(off, len(off)).mean() for _ in range(10000)]
            cells.append(f"{off.mean():+.1f} ({np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f})")
        lines.append(f"| {T:g} K | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def matched_temperature(a: pd.DataFrame) -> str:
    amp = {T: (d["resid_median"].median() / RESID_TO_SIGMA) ** 2 for T, d in a.groupby("T_K")}
    (t0, u0), (t1, u1) = sorted(amp.items())
    slope = (u1 - u0) / (t1 - t0)
    rows = []
    for label, target in (("Debye model, θ_D = 2230 K", 1.52e-3),
                          ("X-ray Debye–Waller, lower", 1.9e-3),
                          ("X-ray Debye–Waller, upper", 2.3e-3)):
        rows.append(f"| {label} | {target:.2e} | {np.sqrt(target):.3f} | {t0 + (target - u0) / slope:.0f} K |")
    return (f"Measured in the simulations: ⟨u_x²⟩ = {u0:.2e} Å² at {t0:g} K and {u1:.2e} Å² at {t1:g} K "
            f"(classical lattice, so linear in temperature).\n\n"
            "| target for real diamond at 300 K | ⟨u_x²⟩ [Å²] | 1D amplitude [Å] | classical lattice temperature that matches it |\n"
            "|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            "The target values are quoted from memory of the literature and should be checked "
            "against a primary source before a temperature is fixed for publication.\n")


def figure(a: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True, sharex=True)
    series = ((300.0, rep.S1, "300 K classical lattice"), (1100.0, rep.S2, "1100 K classical lattice"))
    for ax, col, ylabel, title in (
            (axes[0], "vac_r10", "vacancies within 10 nm of the stopped Sn", "Damage around the stopped ion"),
            (axes[1], "n_vac", "vacancies along the whole track", "Damage along the whole track")):
        for T, colour, label in series:
            d = a[a["T_K"] == T]
            ax.scatter(d["depth_stop_nm"], d[col], s=46, color=colour, alpha=0.85,
                       edgecolors=rep.SURFACE, linewidths=1.2, label=f"{label}, n={len(d)}", zorder=3)
        ax.set_xlabel("Sn stop depth [nm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_xlim(left=0); ax.set_ylim(bottom=-6)
        ax.grid(axis="y", color=rep.RULE, linewidth=0.8, zorder=0)
    axes[0].legend(loc="upper right", fontsize=10)
    fig.suptitle("Sn 35 keV → diamond ⟨110⟩, 0.5° tilt: stronger lattice vibration lets fewer ions "
                 "reach depth; damage at a given depth barely changes", x=0.01, ha="left", fontsize=12.5)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/final"))
    ap.add_argument("--angle", type=float, default=0.5)
    args = ap.parse_args()

    maps, sweep = read(args.maps), read(args.sweep)
    a = sweep[sweep["angle_deg"] == args.angle]
    args.out.mkdir(parents=True, exist_ok=True)
    figure(a, args.out / "temperature_damage.png")
    q_text, _ = quench_check(maps, sweep)
    text = ("## Supplementary checks (random-entry cells, Sn 35 keV, ⟨110⟩)\n\n"
            "### 1. Quench length: 200 ps against 50 ps\n\n" + q_text +
            f"\n### 2. Lattice vibration: 300 K against 1100 K at {args.angle:g}° tilt\n\n" + temperature_table(a, maps) +
            "\n### 3. Which classical temperature matches real diamond at 300 K\n\n" + matched_temperature(a) +
            "\n### Registration quality of these runs\n\n"
            "| lattice | residual [Å] | largest slice step [Å] | false vacancies in pristine control (max) |\n|---|---|---|---|\n" +
            "\n".join(f"| {T:g} K | {d['resid_median'].median():.3f} | {d['reg_max_step'].max():.3f} | "
                      f"{int(d[d['depth_stop_nm'] > 50]['control_vac'].max())} |" for T, d in sweep.groupby("T_K")) + "\n")
    (args.out / "supplementary_checks.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
