"""Generate a bonded LAMMPS data file from an equilibrated LJ snapshot.

Workflow (used by run.py):

1. LAMMPS equilibrates a bondless LJ liquid and writes a restart/data file
   with the final positions.
2. `read_lammps_data` parses that file into positions + box.
3. `select_random_bond_pairs` samples `k` unique pairs whose minimum-image
   distance is below `max_distance`.
4. `write_bonded_data` emits a new LAMMPS data file with the bond topology,
   bond coefficients, and atom_style=bond formatting.

Run separately to inspect the generated topology, or imported by run.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_lammps_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Parse atom positions, types, and box edges from a LAMMPS data file.

    Returns (positions (N,3), atom_types (N,), box_lengths (3,), mass_type_1).
    Assumes ``atom_style atomic`` (id type x y z) which is what the default
    LAMMPS data write after the bondless run produces.
    """
    lines = Path(path).read_text().splitlines()
    n_atoms = 0
    box = np.zeros(3, dtype=np.float64)
    mass = 1.0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.endswith("atoms") and stripped.split()[0].isdigit():
            n_atoms = int(stripped.split()[0])
        if stripped.endswith("xhi"):
            parts = stripped.split()
            box[0] = float(parts[1]) - float(parts[0])
        if stripped.endswith("yhi"):
            parts = stripped.split()
            box[1] = float(parts[1]) - float(parts[0])
        if stripped.endswith("zhi"):
            parts = stripped.split()
            box[2] = float(parts[1]) - float(parts[0])
        if stripped == "Masses":
            # "Masses" then blank line, then lines like "1 39.948"
            j = i + 2
            mass = float(lines[j].split()[1])
        if stripped == "Atoms" or stripped.startswith("Atoms "):
            j = i + 2  # skip blank line
            positions = np.empty((n_atoms, 3), dtype=np.float64)
            atom_types = np.empty(n_atoms, dtype=np.int64)
            for k in range(n_atoms):
                parts = lines[j + k].split()
                aid = int(parts[0]) - 1
                atype = int(parts[1])
                # atom_style atomic: id type x y z
                # atom_style bond:  id mol type x y z (we handle both)
                if len(parts) >= 6:
                    # bond format
                    atype = int(parts[2])
                    x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                else:
                    x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                positions[aid] = (x, y, z)
                atom_types[aid] = atype
            return positions, atom_types, box, mass
        i += 1
    raise RuntimeError(f"Could not parse atoms from {path}")


def select_random_bond_pairs(
    positions: np.ndarray,
    box: np.ndarray,
    k_bonds: int,
    max_distance: float,
    seed: int,
) -> list[tuple[int, int, float]]:
    """Pick `k_bonds` unique pairs whose min-image distance is < max_distance.

    Returns a list of ``(i, j, r_ij)`` with i < j and r_ij the minimum-image
    separation used as the bond rest length.
    """
    if k_bonds == 0:
        return []
    n = positions.shape[0]
    candidates: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[j] - positions[i]
            dr -= box * np.round(dr / box)
            r = float(np.linalg.norm(dr))
            if r < max_distance:
                candidates.append((i, j, r))
    if not candidates:
        raise RuntimeError(
            f"No pairs within {max_distance} — is the box scale right?"
        )
    if k_bonds > len(candidates):
        raise ValueError(
            f"requested {k_bonds} bonds, only {len(candidates)} candidate pairs"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=k_bonds, replace=False)
    return [candidates[c] for c in chosen]


def write_bonded_data(
    path: Path,
    positions: np.ndarray,
    atom_types: np.ndarray,
    box: np.ndarray,
    mass_by_type: dict[int, float],
    bonds: list[tuple[int, int, float]],
    bond_stiffness: float,
) -> None:
    """Emit a LAMMPS data file with atom_style=bond and one harmonic bond per pair.

    Each pair gets its own bond_type so r0 can differ per bond (taken at
    insertion time, per spec). Atoms are assigned molecule IDs by connected
    components of the bond graph (isolated atoms become singleton molecules).
    """
    n = positions.shape[0]
    # Union-find for connected components.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j, _ in bonds:
        union(i, j)
    mol_ids = np.array([find(i) for i in range(n)], dtype=np.int64)
    # Compact molecule IDs to 1..M.
    _, mol_ids = np.unique(mol_ids, return_inverse=True)
    mol_ids = mol_ids + 1

    unique_types = sorted(mass_by_type)
    with open(path, "w") as fh:
        fh.write("LAMMPS data file (bonded LJ, generated by topology.py)\n\n")
        fh.write(f"{n} atoms\n")
        fh.write(f"{len(bonds)} bonds\n")
        fh.write("0 angles\n0 dihedrals\n0 impropers\n\n")
        fh.write(f"{len(unique_types)} atom types\n")
        fh.write(f"{max(1, len(bonds))} bond types\n")  # at least one bond type
        fh.write("0 angle types\n0 dihedral types\n0 improper types\n\n")
        fh.write(f"0.0 {box[0]:.6f} xlo xhi\n")
        fh.write(f"0.0 {box[1]:.6f} ylo yhi\n")
        fh.write(f"0.0 {box[2]:.6f} zlo zhi\n\n")
        fh.write("Masses\n\n")
        for t in unique_types:
            fh.write(f"{t} {mass_by_type[t]:.4f}\n")
        fh.write("\n")
        if bonds:
            fh.write("Bond Coeffs\n\n")
            for b_idx, (_, _, r0) in enumerate(bonds, start=1):
                fh.write(f"{b_idx} {bond_stiffness:.6f} {r0:.6f}\n")
            fh.write("\n")
        fh.write("Atoms\n\n")
        for a_idx in range(n):
            x, y, z = positions[a_idx]
            mol = int(mol_ids[a_idx])
            atype = int(atom_types[a_idx])
            # atom_style bond: id mol type x y z
            fh.write(f"{a_idx + 1} {mol} {atype} {x:.6f} {y:.6f} {z:.6f}\n")
        fh.write("\n")
        if bonds:
            fh.write("Bonds\n\n")
            for b_idx, (i, j, _) in enumerate(bonds, start=1):
                fh.write(f"{b_idx} {b_idx} {i + 1} {j + 1}\n")
            fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equilibrated-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k-bonds", type=int, required=True)
    parser.add_argument("--max-distance", type=float, required=True,
                        help="Cutoff for valid bond pairs (same units as positions, usually Å).")
    parser.add_argument("--bond-stiffness", type=float, required=True,
                        help="LAMMPS harmonic K (energy / length^2).")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    positions, atom_types, box, mass = read_lammps_data(args.equilibrated_data)
    bonds = select_random_bond_pairs(
        positions, box, args.k_bonds, args.max_distance, args.seed,
    )
    mass_by_type = {int(t): float(mass) for t in np.unique(atom_types)}
    write_bonded_data(
        args.output, positions, atom_types, box, mass_by_type,
        bonds, args.bond_stiffness,
    )
    print(f"wrote {len(bonds)} bonds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
