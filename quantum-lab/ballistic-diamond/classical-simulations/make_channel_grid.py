#!/usr/bin/env python3
"""Generate the entry-point grid for a channel-map campaign.

Emits one CSV of `x0_lat,y0_lat` (LAMMPS lattice units) per orientation:
an N x N grid tiling the primitive cell of the projected lattice, centred
on lattice position (10, 10) in the slab.  Because the grid is uniform
over the cell it is area-stratified, so the same runs give an unbiased
depth histogram as well as the map.

    python make_channel_grid.py --orientation 110 --n 7 --out grids/
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

A = 3.5678

# Box axes per orientation, as row vectors in cubic-crystal coordinates,
# matching the `lattice ... orient` lines in simulate.lmp.
FRAMES = {
    "100": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "110": [[1, 1, 0], [0, 0, -1], [-1, 1, 0]],
    "111": [[1, -1, 0], [1, 1, -2], [1, 1, 1]],
}
# LAMMPS lattice spacings = extent of the rotated conventional cell.
SPACING = {
    "100": (A, A, A),
    "110": (A * np.sqrt(2), A, A * np.sqrt(2)),
    "111": (A * np.sqrt(2), 4 * A / np.sqrt(6), A * np.sqrt(3)),
}


def projected_lattice(orientation: str) -> np.ndarray:
    """Primitive 2D vectors of the projection along the beam (box z)."""
    R = np.array(FRAMES[orientation], float)
    R = R / np.linalg.norm(R, axis=1)[:, None]
    prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * A
    gen = (prim @ R.T)[:, :2]
    cand = [c @ gen for c in itertools.product(range(-3, 4), repeat=3)]
    cand = np.array([v for v in cand if np.linalg.norm(v) > 1e-6])
    cand = cand[np.argsort(np.linalg.norm(cand, axis=1))]
    v1 = cand[0]
    for v in cand[1:]:
        if abs(v1[0] * v[1] - v1[1] * v[0]) > 1e-6:
            return np.array([v1, v])
    raise RuntimeError("degenerate projected lattice")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orientation", default="110", choices=sorted(FRAMES))
    ap.add_argument("--n", type=int, default=7, help="grid is n x n over the cell")
    ap.add_argument("--centre", type=float, nargs=2, default=(10.0, 10.0),
                    help="cell centre in lattice units")
    ap.add_argument("--out", type=Path, default=Path("grids"))
    args = ap.parse_args()

    lat = projected_lattice(args.orientation)
    xs, ys, _ = SPACING[args.orientation]
    cx, cy = args.centre

    rows = []
    for i, j in itertools.product(range(args.n), repeat=2):
        p = (i / args.n) * lat[0] + (j / args.n) * lat[1]
        rows.append((cx + p[0] / xs, cy + p[1] / ys))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"grid-{args.orientation}-{args.n}x{args.n}.csv"
    with path.open("w") as fh:
        for x, y in rows:
            fh.write(f"{x:.6f},{y:.6f}\n")

    area = abs(lat[0, 0] * lat[1, 1] - lat[0, 1] * lat[1, 0])
    print(f"orientation {args.orientation}: cell |v1|={np.linalg.norm(lat[0]):.3f} "
          f"|v2|={np.linalg.norm(lat[1]):.3f} A, area={area:.3f} A^2")
    print(f"  {len(rows)} points, in-cell spacing ~{np.sqrt(area)/args.n:.3f} A")
    print(f"  x0_lat range {min(r[0] for r in rows):.3f}..{max(r[0] for r in rows):.3f}, "
          f"y0_lat range {min(r[1] for r in rows):.3f}..{max(r[1] for r in rows):.3f}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
