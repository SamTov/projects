#!/usr/bin/env python3
"""Example end-to-end analysis from a sweep summary HDF5.

Assumes you've already run `ingest.py` and have e.g. sweep-summary.h5
sitting somewhere local.  Generates a small grid of diagnostic plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from ballistic_analysis import load_summary, viz


def main(summary_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(summary_path)
    print(f"Loaded {summary.size} rows  ({(summary['ok']==1).sum()} ok)")

    # --- 1. Depth distribution for each species, at E=35 keV, T=300 K,
    #         stratified by tilt angle.
    for sp in ("sn", "pb"):
        if not (summary["species"] == sp).any():
            continue
        fig = viz.depth_histogram(summary, sp, by="angle_deg",
                                  E_keV=35.0, T_K=300.0)
        fig.savefig(out_dir / f"depth-hist-{sp}-E35-T300.png", dpi=150)
        plt.close(fig)

    # --- 2. Median depth vs tilt angle, grouped by temperature.
    for sp in ("sn", "pb"):
        if not (summary["species"] == sp).any():
            continue
        for E in (20.0, 35.0, 60.0):
            try:
                fig = viz.depth_vs_parameter(summary, sp, x="angle_deg",
                                             group="T_K", E_keV=E)
                fig.savefig(out_dir / f"depth-vs-angle-{sp}-E{int(E)}.png", dpi=150)
                plt.close(fig)
            except ValueError:
                continue

    # --- 3. Channeling fraction vs angle, threshold = 200 A (tweak to taste).
    for sp in ("sn", "pb"):
        if not (summary["species"] == sp).any():
            continue
        fig = viz.channeling_fraction(summary, sp, threshold_A=200.0)
        fig.savefig(out_dir / f"channeling-fraction-{sp}.png", dpi=150)
        plt.close(fig)

    # --- 4. Print per-cell summary table.
    cells = viz.summarize_cells(summary)
    n_cells = len(cells["species"])
    print()
    print(f"{'sp':<3} {'E':>4} {'A':>5} {'T':>5} {'n':>4}  "
          f"{'mean':>7} {'median':>7} {'std':>6}  {'p10':>7} {'p90':>7}")
    for i in range(n_cells):
        print(f"{cells['species'][i]:<3} {cells['E_keV'][i]:>4.0f} "
              f"{cells['angle_deg'][i]:>5.2f} {cells['T_K'][i]:>5.0f} "
              f"{cells['n'][i]:>4d}  "
              f"{cells['depth_mean'][i]:>7.1f} {cells['depth_median'][i]:>7.1f} "
              f"{cells['depth_std'][i]:>6.1f}  "
              f"{cells['depth_p10'][i]:>7.1f} {cells['depth_p90'][i]:>7.1f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path, help="Path to sweep-summary.h5")
    p.add_argument("--out", type=Path, default=Path("./figures"),
                   help="Directory for output PNGs (default: ./figures)")
    args = p.parse_args()
    main(args.summary, args.out)
