"""Build a LAMMPS ``atom_style charge`` data file for molten NaCl.

Places `n_pairs` Na/Cl ion pairs on a B1 (rock-salt) lattice inside a
cubic cell sized for the experimental molten-salt density. Each Na/Cl
carries ±1 e; the full ionic force field (Tosi–Fumi Born–Mayer) is
declared in the LAMMPS input, not the data file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

M_NA = 22.9898   # g/mol
M_CL = 35.4527   # g/mol
Q_NA = +1.0
Q_CL = -1.0


def box_length_from_density(n_pairs: int, density_gpcm3: float = 1.540) -> float:
    """Cubic box edge (Å) for `n_pairs` NaCl at the target mass density."""
    na = 6.02214076e23
    mass_g = n_pairs * (M_NA + M_CL) / na
    volume_cm3 = mass_g / density_gpcm3
    volume_A3 = volume_cm3 * 1e24
    return float(volume_A3 ** (1.0 / 3.0))


def build_nacl_data(
    path: Path, n_pairs: int, box_length: float, seed: int,
) -> None:
    """Write a LAMMPS data file with Na (type 1) / Cl (type 2) on a B1 lattice."""
    # B1 = two interpenetrating FCC sub-lattices. Smallest cubic cell has
    # 4 Na + 4 Cl = 4 pairs. We need at least ceil((n_pairs/4)^(1/3)) cells.
    per_side = max(1, int(np.ceil((n_pairs / 4.0) ** (1.0 / 3.0))))
    a = box_length / per_side  # lattice constant
    half = 0.5 * a

    fcc_sub = np.array([
        [0.0, 0.0, 0.0],
        [0.0, half, half],
        [half, 0.0, half],
        [half, half, 0.0],
    ])
    na_positions = []
    cl_positions = []
    for ix in range(per_side):
        for iy in range(per_side):
            for iz in range(per_side):
                shift = np.array([ix * a, iy * a, iz * a])
                for s in fcc_sub:
                    na_positions.append(s + shift)
                    cl_positions.append(s + shift + np.array([half, 0.0, 0.0]))
    na_positions = np.array(na_positions)
    cl_positions = np.array(cl_positions)
    # Keep only the first n_pairs ions (seed permutation for deterministic thinning).
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(na_positions))[:n_pairs]
    na_positions = na_positions[idx]
    cl_positions = cl_positions[idx]
    # Wrap into the requested box.
    na_positions = na_positions % box_length
    cl_positions = cl_positions % box_length

    atoms = []
    for i, p in enumerate(na_positions):
        atoms.append((i + 1, 1, Q_NA, *p.tolist()))
    for i, p in enumerate(cl_positions):
        atoms.append((i + 1 + n_pairs, 2, Q_CL, *p.tolist()))

    with open(path, "w") as fh:
        fh.write("LAMMPS data file for molten NaCl\n\n")
        fh.write(f"{len(atoms)} atoms\n")
        fh.write("0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n")
        fh.write("2 atom types\n0 bond types\n0 angle types\n")
        fh.write("0 dihedral types\n0 improper types\n\n")
        fh.write(f"0.0 {box_length:.6f} xlo xhi\n")
        fh.write(f"0.0 {box_length:.6f} ylo yhi\n")
        fh.write(f"0.0 {box_length:.6f} zlo zhi\n\n")
        fh.write(f"Masses\n\n1 {M_NA}\n2 {M_CL}\n\n")
        fh.write("Atoms\n\n")
        for aid, at_type, q, x, y, z in atoms:
            fh.write(f"{aid} {at_type} {q:.4f} {x:.6f} {y:.6f} {z:.6f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-pairs", type=int, required=True)
    parser.add_argument("--density", type=float, default=1.540,
                        help="g/cm^3 (molten NaCl at 1100 K)")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    box = box_length_from_density(args.n_pairs, args.density)
    build_nacl_data(args.output, args.n_pairs, box, args.seed)
    print(f"wrote {args.n_pairs} NaCl pairs to {args.output} (box = {box:.3f} Å)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
