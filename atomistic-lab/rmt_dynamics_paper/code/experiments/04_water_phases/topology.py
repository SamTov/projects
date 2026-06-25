"""Build a LAMMPS data file for TIP4P/2005 water.

Places `n_water` molecules on a cubic lattice with random orientations at
approximately liquid density, writes out an atom_style=full data file with
O-H bonds and H-O-H angles so LAMMPS + SHAKE can keep the geometry rigid.

The massless M site is synthesised by the ``lj/cut/tip4p/long`` pair style
at run time and is NOT present in the data file.

For the ice Ih endpoint, the spec calls for a proton-disordered starting
configuration. Building one from scratch is non-trivial; the recommended
workflow is to render a standard ice Ih `.data` file with an external tool
(e.g. genice, packmol, or VMD) and point run.py at it via ``--ice-data``.

Constants
---------
O-H bond length       : 0.9572 Å  (TIP4P/2005 rigid geometry)
H-O-H angle           : 104.52°
Target liquid density : 0.9970 g/cm³ (298 K, ~experimental)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OH_BOND = 0.9572     # Å
HOH_ANGLE = 104.52   # degrees
M_WATER = 18.0154    # g/mol

# TIP4P/2005 partial charges. Assigned to types (1 = O, 2 = H), with a
# compensating charge at the massless M-site handled by the pair style.
Q_O = 0.0
Q_H = 0.5564


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3x3 rotation matrix (Shoemaker's quaternion method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _water_local_coords() -> np.ndarray:
    """Oxygen at origin, two hydrogens in the xy-plane."""
    half = np.deg2rad(HOH_ANGLE / 2.0)
    h1 = np.array([OH_BOND * np.sin(half),  OH_BOND * np.cos(half), 0.0])
    h2 = np.array([-OH_BOND * np.sin(half), OH_BOND * np.cos(half), 0.0])
    return np.array([[0.0, 0.0, 0.0], h1, h2])


def build_water_data(
    path: Path, n_water: int, box_length: float, seed: int,
) -> None:
    """Write a LAMMPS ``atom_style full`` data file for n_water TIP4P/2005 molecules."""
    rng = np.random.default_rng(seed)
    # Cubic lattice sized to accommodate n_water with spacing derived from box.
    per_side = int(np.ceil(n_water ** (1.0 / 3.0)))
    spacing = box_length / per_side
    sites = _water_local_coords()

    atoms: list[tuple[int, int, int, float, float, float, float]] = []  # (id, mol, type, q, x, y, z)
    bonds: list[tuple[int, int, int, int]] = []  # (id, type, i, j)
    angles: list[tuple[int, int, int, int, int]] = []  # (id, type, i, j, k)

    placed = 0
    for ix in range(per_side):
        for iy in range(per_side):
            for iz in range(per_side):
                if placed >= n_water:
                    break
                centre = np.array([
                    (ix + 0.5) * spacing,
                    (iy + 0.5) * spacing,
                    (iz + 0.5) * spacing,
                ])
                R = _random_rotation(rng)
                coords = (sites @ R.T) + centre  # (3, 3): O, H1, H2
                mol_id = placed + 1
                o_id = 3 * placed + 1
                h1_id = 3 * placed + 2
                h2_id = 3 * placed + 3
                atoms.append((o_id, mol_id, 1, Q_O, *coords[0].tolist()))
                atoms.append((h1_id, mol_id, 2, Q_H, *coords[1].tolist()))
                atoms.append((h2_id, mol_id, 2, Q_H, *coords[2].tolist()))
                bonds.append((2 * placed + 1, 1, o_id, h1_id))
                bonds.append((2 * placed + 2, 1, o_id, h2_id))
                angles.append((placed + 1, 1, h1_id, o_id, h2_id))
                placed += 1

    with open(path, "w") as fh:
        fh.write("LAMMPS data file for TIP4P/2005 water\n\n")
        fh.write(f"{len(atoms)} atoms\n")
        fh.write(f"{len(bonds)} bonds\n")
        fh.write(f"{len(angles)} angles\n")
        fh.write("0 dihedrals\n0 impropers\n\n")
        fh.write("2 atom types\n1 bond types\n1 angle types\n")
        fh.write("0 dihedral types\n0 improper types\n\n")
        fh.write(f"0.0 {box_length:.6f} xlo xhi\n")
        fh.write(f"0.0 {box_length:.6f} ylo yhi\n")
        fh.write(f"0.0 {box_length:.6f} zlo zhi\n\n")
        fh.write("Masses\n\n1 15.9994\n2 1.00784\n\n")
        fh.write("Atoms\n\n")
        for aid, mol, at_type, q, x, y, z in atoms:
            fh.write(f"{aid} {mol} {at_type} {q:.4f} {x:.6f} {y:.6f} {z:.6f}\n")
        fh.write("\nBonds\n\n")
        for bid, bt, i, j in bonds:
            fh.write(f"{bid} {bt} {i} {j}\n")
        fh.write("\nAngles\n\n")
        for ang_id, at, i, j, k in angles:
            fh.write(f"{ang_id} {at} {i} {j} {k}\n")


def liquid_box_length(n_water: int, density_gpcm3: float = 0.997) -> float:
    """Cubic box edge (Å) to hit the target density for n_water molecules."""
    na = 6.02214076e23
    volume_A3 = n_water * M_WATER / (density_gpcm3 * na * 1e-24)
    return float(volume_A3 ** (1.0 / 3.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-water", type=int, required=True)
    parser.add_argument("--density", type=float, default=0.997,
                        help="g/cm^3, used if --box-length not given")
    parser.add_argument("--box-length", type=float, default=None,
                        help="Override cubic box edge (Å)")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    box = args.box_length or liquid_box_length(args.n_water, args.density)
    build_water_data(args.output, args.n_water, box, args.seed)
    print(f"wrote {args.n_water} water molecules to {args.output} (box = {box:.3f} Å)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
