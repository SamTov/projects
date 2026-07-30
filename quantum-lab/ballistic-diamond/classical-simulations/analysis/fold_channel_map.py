#!/usr/bin/env python3
"""Depth-scored channel map from randomised-entry runs, by folding.

Our production runs place the ion at a RANDOM (x,y) over a 4x4 lattice-unit
patch rather than on a grid, so they aren't a channel scan.  But every entry
point is recoverable (first frame of collision-ion.lammpstraj), and folding
those points modulo the 2D projected lattice turns the ensemble into a
sparse-but-real map of stop depth vs position within the channel cell.

Two corrections that matter:
  * the ion is launched ~100 A above the surface, so at non-zero tilt it
    drifts laterally before entry -- we extrapolate along the initial
    velocity to the surface plane before folding;
  * the projection frame is the SIMULATION BOX frame (x=[110], y=[00-1],
    beam along z=[-110]), not an arbitrary <110> frame, so the overlay
    registers with the data.
"""
from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ballistic_analysis.reader import read_ion_trajectory

A = 3.5678
BOND_MAX = 1.7
BASE = "/work/stovey/ballistic-diamond/tersoff-sweep/orient-110/energy-35"
SLAB_TOP_LAT = 670
Z_SPACING = 5.045649           # LAMMPS z spacing for the 110 box
SURFACE = SLAB_TOP_LAT * Z_SPACING

# Simulation box frame: rows are the box x, y, z axes in crystal coordinates.
R_BOX = np.array([
    [1.0,  1.0, 0.0],
    [0.0,  0.0, -1.0],
    [-1.0, 1.0, 0.0],
])
R_BOX = R_BOX / np.linalg.norm(R_BOX, axis=1)[:, None]

_BASIS = np.array([
    [0.00, 0.00, 0.00], [0.00, 0.50, 0.50], [0.50, 0.00, 0.50],
    [0.50, 0.50, 0.00], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
])


def projected_lattice() -> np.ndarray:
    """Two primitive vectors of the projection of the fcc lattice."""
    prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * A
    gen = (prim @ R_BOX.T)[:, :2]
    cand = []
    for c in itertools.product(range(-3, 4), repeat=3):
        v = c @ gen
        if np.linalg.norm(v) > 1e-6:
            cand.append(v)
    cand = np.array(cand)
    cand = cand[np.argsort(np.linalg.norm(cand, axis=1))]
    v1 = cand[0]
    for v in cand[1:]:
        if abs(np.cross(v1, v)) > 1e-6:
            return np.array([v1, v])
    raise RuntimeError("no 2D lattice")


def crystal_overlay(half: float):
    reach = int(np.ceil(half / A)) + 3
    cells = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), float)
    atoms = (cells[:, None, :] + _BASIS[None, :, :]).reshape(-1, 3) * A
    proj = atoms @ R_BOX.T
    keep = (np.abs(proj[:, 2]) < 1.5 * A) & (np.abs(proj[:, 0]) < half + 2) \
        & (np.abs(proj[:, 1]) < half + 2)
    atoms, proj = atoms[keep], proj[keep]
    from scipy.spatial import cKDTree
    pairs = cKDTree(atoms).query_pairs(BOND_MAX, output_type="ndarray")
    segs = np.stack([proj[pairs[:, 0], :2], proj[pairs[:, 1], :2]], axis=1)
    segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 0.15]
    cols = np.unique(np.round(proj[:, :2], 3), axis=0)
    return cols, segs


def collect(temp, angle):
    """Entry point at the surface plane + stop depth, per ensemble."""
    rows = []
    for d in sorted(glob.glob(f"{BASE}/temperature-{temp}/angle-{angle}-*/")):
        tr = read_ion_trajectory(Path(d) / "collision-ion.lammpstraj")
        if tr is None or len(tr["pos"]) < 2:
            continue
        p0, v0 = tr["pos"][0], tr["vel"][0]
        if v0[2] >= 0:
            continue
        t = (p0[2] - SURFACE) / (-v0[2])        # extrapolate to the surface
        entry = p0[:2] + v0[:2] * t
        depth = (SURFACE - float(tr["pos"][-1, 2])) / 10.0
        rows.append((entry[0], entry[1], depth))
    return np.array(rows) if rows else np.zeros((0, 3))


def fold(points, lat):
    """Reduce to the primitive cell, centred on the origin."""
    frac = points @ np.linalg.inv(lat)
    frac -= np.round(frac)
    return frac @ lat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("figures/folded_channel_map"))
    ap.add_argument("--half-width", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    lat = projected_lattice()
    print(f"projected cell |v1|={np.linalg.norm(lat[0]):.3f} |v2|={np.linalg.norm(lat[1]):.3f} A, "
          f"area={abs(np.cross(*lat)):.3f} A^2")

    panels = [(300, "0", "300 K, 0°"), (300, "0.5", "300 K, 0.5°"),
              (300, "2", "300 K, 2°"), (1100, "0.5", "1100 K (DW), 0.5°")]
    data = []
    for temp, angle, label in panels:
        rows = collect(temp, angle)
        if not len(rows):
            print(f"  {label}: no data"); continue
        folded = fold(rows[:, :2], lat)
        data.append((label, folded, rows[:, 2]))
        print(f"  {label}: n={len(rows)}, depth {rows[:,2].min():.0f}-{rows[:,2].max():.0f} nm")

    half = args.half_width
    cols, segs = crystal_overlay(half)
    vmax = max(d[2].max() for d in data)

    fig, axes = plt.subplots(1, len(data), figsize=(3.5 * len(data), 4.0))
    axes = np.atleast_1d(axes)
    for k, (ax, (label, pts, dep)) in enumerate(zip(axes, data)):
        ax.add_collection(LineCollection(segs, colors="0.25", linewidths=0.8,
                                         alpha=0.6, zorder=1))
        m = (np.abs(cols[:, 0]) <= half) & (np.abs(cols[:, 1]) <= half)
        ax.scatter(cols[m, 0], cols[m, 1], s=22, c="0.1", zorder=2,
                   edgecolors="white", linewidths=0.5)
        # replicate the folded points across the view
        reps = int(np.ceil(2 * half / np.linalg.norm(lat[0]))) + 1
        for i, j in itertools.product(range(-reps, reps + 1), repeat=2):
            sh = pts + i * lat[0] + j * lat[1]
            sel = (np.abs(sh[:, 0]) <= half) & (np.abs(sh[:, 1]) <= half)
            if sel.any():
                sc = ax.scatter(sh[sel, 0], sh[sel, 1], c=dep[sel], cmap="plasma",
                                vmin=0, vmax=vmax, s=95, zorder=3,
                                edgecolors="white", linewidths=0.8)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_title(f"{label}  (n={len(dep)})", fontsize=10)
        ax.set_xlabel("entry x [Å]")
        ax.set_ylabel("entry y [Å]" if k == 0 else "")
        if k:
            ax.set_yticklabels([])
        ax.text(-0.12, 1.05, "abcd"[k], transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="right")

    cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.022, pad=0.015)
    cb.set_label("Sn stop depth [nm]", fontsize=9)
    fig.suptitle("Folded channel map — Sn 35 keV, ⟨110⟩ — entry point reduced to the "
                 "projected cell (sparse: randomised, not gridded, entry)", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
