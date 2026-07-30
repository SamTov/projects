#!/usr/bin/env python3
"""Publication channel-map figure: damage score vs ion entry position.

Rebuilt from the figure specification in CHANNEL_MAP_REPRODUCTION_RUNBOOK.md
section 8, so it can consume either a colleague's `entry_score_summary.csv`
files or our own scored output.

The distinctive features of this figure (the "special" part):
  * the sampled projected unit cell is tiled PERIODICALLY across the full
    view, so a small scanned domain fills a -3..+3 A window;
  * projected carbon columns and real projected C-C bonds are overlaid, so
    the map is read against the crystal structure;
  * bonds are computed from actual 3D neighbours and then projected -- for
    a [100] view that yields the correct diagonal pattern and never an
    axis-aligned square grid;
  * one shared colour normalisation across all panels.

Input CSV per condition: columns x0, y0, mean_score (n_runs optional).

Usage
-----
    # demo layout with synthetic scores (no data needed)
    python plot_channel_map.py --demo --out figures/channel_map_demo

    # real data
    python plot_channel_map.py \
        --panel "[110] 0 deg:110:path/to/110/entry_score_summary.csv" \
        --panel "[100] 0 deg:100:path/to/100/entry_score_summary.csv" \
        --panel "[100] 7 deg:100:path/to/100_tilt7/entry_score_summary.csv" \
        --out figures/channel_map_2nm
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

A_DIAMOND = 3.5678
BOND_MAX = 1.7          # A, C-C nearest neighbour 1.545
FACET_NORMALS = {"100": (0, 0, 1), "110": (1, 1, 0), "111": (1, 1, 1)}

_BASIS = np.array([
    [0.00, 0.00, 0.00], [0.00, 0.50, 0.50], [0.50, 0.00, 0.50],
    [0.50, 0.50, 0.00], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
])


def view_frame(normal) -> np.ndarray:
    """Orthonormal rows (e1, e2, e3) with e3 along the view axis."""
    e3 = np.asarray(normal, float)
    e3 /= np.linalg.norm(e3)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(e3 @ seed) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    e1 = seed - (seed @ e3) * e3
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return np.array([e1, e2, e3])


def crystal_projection(facet: str, half_width: float, a: float = A_DIAMOND):
    """Projected column positions and projected C-C bond segments.

    Bonds are found in 3D then projected, so the drawn pattern is the true
    projection of the bond network rather than a guessed 2D grid.
    """
    R = view_frame(FACET_NORMALS[facet])
    reach = int(np.ceil(half_width / a)) + 3
    cells = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), float)
    atoms = (cells[:, None, :] + _BASIS[None, :, :]).reshape(-1, 3) * a
    proj = atoms @ R.T                       # columns: (e1, e2, view)

    # keep a slab a few cells thick along the view axis -- enough for bonds,
    # cheap enough to draw
    keep = np.abs(proj[:, 2]) < 1.5 * a
    atoms, proj = atoms[keep], proj[keep]
    inview = (np.abs(proj[:, 0]) < half_width + 2) & (np.abs(proj[:, 1]) < half_width + 2)
    atoms, proj = atoms[inview], proj[inview]

    from scipy.spatial import cKDTree
    pairs = cKDTree(atoms).query_pairs(BOND_MAX, output_type="ndarray")
    segs = np.stack([proj[pairs[:, 0], :2], proj[pairs[:, 1], :2]], axis=1)
    # drop bonds that project to (almost) a point -- they lie along the view axis
    segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 0.15]

    cols = np.unique(np.round(proj[:, :2], 3), axis=0)
    return cols, segs


def projected_lattice(facet: str, a: float = A_DIAMOND) -> np.ndarray:
    """Primitive 2D vectors of the projected lattice (for periodic tiling)."""
    R = view_frame(FACET_NORMALS[facet])
    prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    gen = (prim @ R.T)[:, :2]
    cand = []
    for coeffs in itertools.product(range(-3, 4), repeat=3):
        v = coeffs @ gen
        if np.linalg.norm(v) > 1e-6:
            cand.append(v)
    cand = np.array(cand)
    cand = cand[np.argsort(np.linalg.norm(cand, axis=1))]
    v1 = cand[0]
    for v in cand[1:]:
        if abs(np.cross(v1, v)) > 1e-6:
            return np.array([v1, v])
    raise RuntimeError(f"could not build 2D lattice for facet {facet}")


def tile(points: np.ndarray, values: np.ndarray, facet: str, half_width: float):
    """Replicate the sampled cell periodically to fill the view."""
    v1, v2 = projected_lattice(facet)
    reps = int(np.ceil(2 * half_width / min(np.linalg.norm(v1), np.linalg.norm(v2)))) + 2
    out_p, out_v = [], []
    for i, j in itertools.product(range(-reps, reps + 1), repeat=2):
        shifted = points + i * v1 + j * v2
        m = ((np.abs(shifted[:, 0]) < half_width + 1.0)
             & (np.abs(shifted[:, 1]) < half_width + 1.0))
        if m.any():
            out_p.append(shifted[m])
            out_v.append(values[m])
    return np.vstack(out_p), np.concatenate(out_v)


def read_panel(path: Path):
    xs, ys, sc = [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x0"]))
            ys.append(float(row["y0"]))
            sc.append(float(row["mean_score"]))
    return np.column_stack([xs, ys]), np.array(sc)


def demo_panel(facet: str, level: float, seed: int):
    """Synthetic scores: damage is high on the atomic columns, low in the
    open channel. For LAYOUT REVIEW ONLY -- not physical output."""
    rng = np.random.default_rng(seed)
    v1, v2 = projected_lattice(facet)
    fr = np.linspace(-0.5, 0.5, 11)
    pts = np.array([f1 * v1 + f2 * v2 for f1 in fr for f2 in fr])
    cols, _ = crystal_projection(facet, 4.0)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(cols).query(pts)
    shape = np.exp(-(d ** 2) / (2 * 0.55 ** 2)) + 0.10
    sc = shape * (level / shape.mean())      # cell mean == the quoted level
    return pts, sc * rng.normal(1.0, 0.06, len(sc))


def draw(ax, points, scores, facet, half_width, vmin, vmax, cmap):
    p, s = tile(points, scores, facet, half_width)
    grid = np.linspace(-half_width, half_width, 400)
    gx, gy = np.meshgrid(grid, grid)
    from scipy.interpolate import griddata
    field = griddata(p, s, (gx, gy), method="linear")
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[-half_width, half_width, -half_width, half_width],
                   interpolation="bilinear", zorder=1)
    cols, segs = crystal_projection(facet, half_width)
    ax.add_collection(LineCollection(segs, colors="0.15", linewidths=0.8,
                                     alpha=0.55, zorder=2))
    m = (np.abs(cols[:, 0]) <= half_width) & (np.abs(cols[:, 1]) <= half_width)
    ax.scatter(cols[m, 0], cols[m, 1], s=16, c="0.05", zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    ax.set_xlabel("entry x [Å]")
    return im


def build(panels, out_stem: Path, vmin, vmax, cmap, half_width, layout, dpi, title=None):
    n = len(panels)
    if layout == "horizontal":
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.7))
    else:
        fig, axes = plt.subplots(n, 1, figsize=(3.9, 3.5 * n))
    axes = np.atleast_1d(axes)

    for k, (ax, (label, facet, points, scores)) in enumerate(zip(axes, panels)):
        im = draw(ax, points, scores, facet, half_width, vmin, vmax, cmap)
        ax.set_title(label, fontsize=11, pad=6)
        if (layout == "horizontal" and k == 0) or layout == "vertical":
            ax.set_ylabel("entry y [Å]")
        else:
            ax.set_yticklabels([])
        ax.text(-0.14, 1.06, "abcdefg"[k], transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="right")

    # small colorbar inset in the lower right of the final panel
    cax = axes[-1].inset_axes([0.60, 0.06, 0.34, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("displaced C within 2 nm", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    for spine in ("bottom", "top"):
        cax.spines[spine].set_linewidth(0.4)

    if title:
        fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [f"{out_stem}.{e}" for e in ("png", "pdf", "svg")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", action="append", default=[],
                    help="LABEL:FACET:CSV  (repeatable, in panel order)")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic scores for layout review (not physics)")
    ap.add_argument("--out", type=Path, default=Path("figures/channel_map"))
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=220.0)
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--half-width", type=float, default=3.0, help="view half-width [Å]")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    if args.demo:
        spec = [("⟨110⟩, 0°", "110", 58.0), ("⟨100⟩, 0°", "100", 79.0),
                ("⟨100⟩, 7°", "100", 94.0)]
        panels = [(lab, f, *demo_panel(f, lvl, i)) for i, (lab, f, lvl) in enumerate(spec)]
        title = "SYNTHETIC DATA — layout review only"
    else:
        if not args.panel:
            ap.error("give --panel LABEL:FACET:CSV (or --demo)")
        panels, title = [], None
        for spec in args.panel:
            label, facet, path = spec.split(":", 2)
            pts, sc = read_panel(Path(path))
            panels.append((label, facet, pts, sc))

    written = []
    for layout in ("horizontal", "vertical"):
        written += build(panels, Path(f"{args.out}_{layout}"), args.vmin, args.vmax,
                         args.cmap, args.half_width, layout, args.dpi, title)
    for w in written:
        print("wrote", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
