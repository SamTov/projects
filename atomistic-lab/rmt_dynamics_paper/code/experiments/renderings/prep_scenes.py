"""Build scene files (positions + colour classes) for the ZnVis renderings.

Run with the rmt_dynamics environment; writes one .npz per scene into
<out>. render_znvis.py (ZnVis environment) turns them into images.

Scenes
------
bonded_liquid : last frame of the long-window bond-addition run (27 bonds),
                atoms coloured by their weight in the sub-bulk eigenvectors
                of C, bonds drawn as cylinders.
lj_<phase>    : short 20 atm LJ runs (crystal 40 K, liquid 100 K,
                vapour 150 K) written with positions for snapshots.

Usage: python prep_scenes.py <correlation_base> <out_dir> <lmp_command>
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import parse_lammps_dump_custom, render_template  # noqa: E402
from rmt_dynamics import load_C  # noqa: E402

SIGMA = 3.405


def last_frame(dump_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and box of the final frame without loading the whole dump."""
    with open(dump_path) as fh:
        lines = fh.readlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("ITEM: TIMESTEP")]
    tail = Path(dump_path.parent / "_last_frame.lammpstrj")
    tail.write_text("".join(lines[starts[-1]:]))
    d = parse_lammps_dump_custom(tail)
    tail.unlink()
    return d.positions[0], d.box


def bonded_liquid(base: Path, out: Path, k: int = 27, seed: int = 1) -> None:
    run = base / "02_bond_addition_long" / "runs" / f"k_bonds={k}_seed={seed}"
    C, _ = load_C(run / "C.npz")
    mu, Phi = np.linalg.eigh(C)
    # Mode 0 is the momentum zero mode; the next k are the bond modes.
    weight = np.sum(Phi[:, 1:k + 1] ** 2, axis=1)
    pos, box = last_frame(run / "dump.lammpstrj")
    bonds = np.load(run / "bonds.npy")
    pos = np.mod(pos, box)
    a, b = pos[bonds[:, 0]], pos[bonds[:, 1]]
    d = b - a
    d -= box * np.round(d / box)
    mid = np.mod(a + 0.5 * d, box)
    # Draw both atoms of a bond on the same side of the box as the midpoint.
    a_draw, b_draw = mid - 0.5 * d, mid + 0.5 * d
    pos[bonds[:, 0]], pos[bonds[:, 1]] = a_draw, b_draw
    np.savez(out / "bonded_liquid.npz", positions=pos, box=box, weight=weight,
             bond_mid=mid, bond_vec=d, bonded=np.isin(np.arange(len(pos)), bonds))
    print("bonded_liquid", len(pos), "atoms;", len(bonds), "bonds; weight on bonded atoms",
          weight[np.unique(bonds)].mean().round(3), "vs others", weight[~np.isin(np.arange(len(pos)), bonds)].mean().round(4))


def lj_snapshots(out: Path, lmp: str) -> None:
    work = out / "lj_runs"
    work.mkdir(exist_ok=True)
    cases = {"crystal": ("fcc", 40.0, 1.0), "liquid": ("dense", 100.0, 0.8), "vapour": ("dilute", 150.0, 0.02)}
    for name, (init, T, rho) in cases.items():
        d = work / name
        d.mkdir(exist_ok=True)
        ctx = {"seed": 7, "n_atoms": 500, "temperature": T, "pressure": 20.0, "timestep": 2.0,
               "n_equil": 25000, "n_prod": 1, "sample_every": 1, "out_dump": "snap.lammpstrj",
               "dump_columns": "x y z", "ensemble": "npt", "mass": 39.948, "sigma": SIGMA,
               "epsilon": 0.2381, "cutoff": 10.215, "init": init,
               "box_length": (500 / (rho / SIGMA ** 3)) ** (1 / 3),
               "n_cells": 5, "lattice_a": SIGMA * (4.0 / rho) ** (1 / 3)}
        (d / "in.lammps").write_text(render_template("lj_nvt.in.j2", ctx))
        subprocess.run([*shlex.split(lmp), "-log", "none", "-in", "in.lammps"], cwd=d, check=True,
                       stdout=open(d / "log.lammps", "w"), stderr=subprocess.STDOUT)
        pos, box = last_frame(d / "snap.lammpstrj")
        np.savez(out / f"lj_{name}.npz", positions=np.mod(pos, box), box=box)
        print(name, "box", box.round(2))


def main() -> int:
    base, out, lmp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    bonded_liquid(base, out)
    lj_snapshots(out, lmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
