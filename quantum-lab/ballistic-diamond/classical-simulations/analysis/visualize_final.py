"""Visualize a final.data snapshot using ZnVis.

Usage:
    python visualize_final.py /path/to/final.data
    python visualize_final.py /path/to/final.data --neighborhood 50
    python visualize_final.py /path/to/final.data --full       # all 3M carbons (slow!)
    python visualize_final.py /path/to/final.data --no-box

Default behavior is to render the heavy ion as a bright sphere and show only
the carbons within `--neighborhood` Å of it -- that's the damage cluster you
actually want to look at; rendering 3M spheres in Open3D is painful.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Tell Python where ZnVis lives (no install needed if you've cloned it).
sys.path.insert(0, str(Path("/Users/samueltovey/work/repositories/ZnVis").expanduser()))
import znvis as vis


def parse_lammps_data(path: Path):
    """Return (types, positions[N,3], box{xlo,xhi,...}) from a LAMMPS data file."""
    section = None
    types: list[int] = []
    coords: list[list[float]] = []
    box = {}

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            # Box bounds
            if line.endswith("xlo xhi"):
                lo, hi, *_ = line.split()
                box["xlo"], box["xhi"] = float(lo), float(hi)
                continue
            if line.endswith("ylo yhi"):
                lo, hi, *_ = line.split()
                box["ylo"], box["yhi"] = float(lo), float(hi)
                continue
            if line.endswith("zlo zhi"):
                lo, hi, *_ = line.split()
                box["zlo"], box["zhi"] = float(lo), float(hi)
                continue
            # Section markers
            head = line.split("#", 1)[0].strip()
            if head in ("Atoms", "Velocities", "Masses"):
                section = head.lower()
                continue
            if not line or line.startswith("#"):
                continue
            if section != "atoms":
                continue
            parts = line.split()
            # atomic style: id type x y z [ix iy iz]
            if len(parts) < 5:
                continue
            try:
                t = int(parts[1])
                xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
            except ValueError:
                continue
            types.append(t)
            coords.append(xyz)

    return np.array(types), np.array(coords), box


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="Path to final.data")
    p.add_argument("--neighborhood", type=float, default=30.0,
                   help="Show carbons within this many Å of the ion (default 30).")
    p.add_argument("--full", action="store_true",
                   help="Show all ~3M carbons.  WARNING: render is slow / heavy.")
    p.add_argument("--no-box", action="store_true",
                   help="Skip the bounding-box overlay.")
    p.add_argument("--ion-radius", type=float, default=2.0,
                   help="Radius (Å) of the heavy-ion sphere.")
    p.add_argument("--carbon-radius", type=float, default=0.5,
                   help="Radius (Å) of carbon spheres.")
    p.add_argument("--ion-colour", default="auto",
                   help="Hex / name for ion colour ('auto' = red for Sn, "
                        "blue for Pb based on the data file path).")
    args = p.parse_args()

    types, positions, box = parse_lammps_data(args.data)
    print(f"Loaded {len(types)} atoms from {args.data}")
    print(f"  carbons : {(types == 1).sum()}")
    print(f"  ions    : {(types == 2).sum()}")
    print(f"  box     : x[{box.get('xlo'):.1f},{box.get('xhi'):.1f}] "
          f"y[{box.get('ylo'):.1f},{box.get('yhi'):.1f}] "
          f"z[{box.get('zlo'):.1f},{box.get('zhi'):.1f}]")

    # ---- Locate the ion ----
    ion_mask = (types == 2)
    if not ion_mask.any():
        print("ERROR: no type-2 (ion) atom in this data file.", file=sys.stderr)
        return 1
    ion_pos = positions[ion_mask][0]
    print(f"  ion at  : ({ion_pos[0]:.2f}, {ion_pos[1]:.2f}, {ion_pos[2]:.2f}) Å")
    print(f"  depth   : {box['zhi'] - 30.0 - ion_pos[2]:.1f} Å below top surface")

    # ---- Carbons (subset by default) ----
    carbon_pos = positions[types == 1]
    if not args.full:
        d = np.linalg.norm(carbon_pos - ion_pos, axis=1)
        carbon_pos = carbon_pos[d < args.neighborhood]
        print(f"  showing {len(carbon_pos)} carbons within "
              f"{args.neighborhood:g} Å of the ion (use --full for all)")

    # ---- Decide ion colour ----
    if args.ion_colour == "auto":
        ion_rgb = (np.array([255, 60, 60]) / 255 if "tersoff-sweep-pb" not in str(args.data)
                   else np.array([90, 130, 255]) / 255)
    else:
        # Accept #RRGGBB or a named tuple "r,g,b" in 0-255.
        s = args.ion_colour.lstrip("#")
        if len(s) == 6:
            ion_rgb = np.array([int(s[i:i+2], 16) for i in (0, 2, 4)]) / 255
        else:
            ion_rgb = np.array([float(c) for c in s.split(",")]) / 255

    # ---- ZnVis particles ----
    carbon_material = vis.Material(colour=np.array([0.55, 0.55, 0.55]), alpha=0.35)
    carbon_mesh = vis.Sphere(radius=args.carbon_radius,
                             material=carbon_material, resolution=4)
    carbon_particle = vis.Particle(name="Carbon",
                                   mesh=carbon_mesh,
                                   position=carbon_pos[None, :, :])

    ion_material = vis.Material(colour=ion_rgb, alpha=1.0)
    ion_mesh = vis.Sphere(radius=args.ion_radius,
                          material=ion_material, resolution=24)
    ion_particle = vis.Particle(name="Ion",
                                mesh=ion_mesh,
                                position=ion_pos.reshape(1, 1, 3))

    # ---- Bounding box ----
    bbox = None
    if not args.no_box:
        center = np.array([
            (box["xlo"] + box["xhi"]) / 2,
            (box["ylo"] + box["yhi"]) / 2,
            (box["zlo"] + box["zhi"]) / 2,
        ])
        size = np.array([
            box["xhi"] - box["xlo"],
            box["yhi"] - box["ylo"],
            box["zhi"] - box["zlo"],
        ])
        bbox = vis.BoundingBox(center=center, box_size=size)

    visualizer = vis.Visualizer(
        particles=[carbon_particle, ion_particle],
        frame_rate=1,
        bounding_box=bbox,
    )
    visualizer.run_visualization()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
