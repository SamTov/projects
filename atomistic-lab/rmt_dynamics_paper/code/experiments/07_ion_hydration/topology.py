"""Water box with monovalent ion pairs for the ion-hydration pilot.

Waters are placed on a cubic lattice with random orientations (the run
melts them at 400 K before cooling); `n_pairs` cations and `n_pairs`
anions then replace randomly chosen waters at their oxygen positions.
Atom order in the data file: all waters (O, H, H per molecule), then
cations, then anions. Types: 1 O, 2 H, 3 cation, 4 anion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OH, HOH = 0.9572, 104.52
Q_O, Q_H = -1.04844, 0.52422          # TIP4P-Ew; O carries the M-site charge
MASS = {1: 15.99943, 2: 1.007947}
ION_MASS = {"Li": 6.94, "Na": 22.99, "K": 39.1, "Cl": 35.45}
M_WATER = 18.01528


def _rotation(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def build(path: Path, n_water: int, n_pairs: int, cation: str | None, anion: str | None,
          density: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_sites = n_water + 2 * n_pairs
    mass_g = (n_water * M_WATER + n_pairs * (ION_MASS.get(cation, 0) + ION_MASS.get(anion, 0))) / 6.02214076e23
    L = (mass_g / density * 1e24) ** (1 / 3)
    per = int(np.ceil(n_sites ** (1 / 3)))
    sp = L / per
    grid = np.array([[(i + .5) * sp, (j + .5) * sp, (k + .5) * sp]
                     for i in range(per) for j in range(per) for k in range(per)])
    grid = grid[rng.permutation(len(grid))[:n_sites]]
    half = np.deg2rad(HOH / 2)
    local = np.array([[0, 0, 0], [OH * np.sin(half), OH * np.cos(half), 0], [-OH * np.sin(half), OH * np.cos(half), 0]])
    water_sites, ion_sites = grid[:n_water], grid[n_water:]
    atoms, bonds, angles = [], [], []
    for m, c in enumerate(water_sites):
        xyz = local @ _rotation(rng).T + c
        base = 3 * m
        for a, (t, q) in enumerate(((1, Q_O), (2, Q_H), (2, Q_H))):
            atoms.append((base + a + 1, m + 1, t, q, *xyz[a]))
        bonds += [(2 * m + 1, base + 1, base + 2), (2 * m + 2, base + 1, base + 3)]
        angles.append((m + 1, base + 2, base + 1, base + 3))
    aid, mol = 3 * n_water, n_water
    for k, c in enumerate(ion_sites):
        t, q = (3, 1.0) if k < n_pairs else (4, -1.0)
        aid += 1; mol += 1
        atoms.append((aid, mol, t, q, *c))
    masses = {1: MASS[1], 2: MASS[2], 3: ION_MASS.get(cation, 22.99), 4: ION_MASS.get(anion, 35.45)}
    with open(path, "w") as fh:
        fh.write(f"Water with {n_pairs} {cation or ''}{anion or ''} pairs\n\n")
        fh.write(f"{len(atoms)} atoms\n{len(bonds)} bonds\n{len(angles)} angles\n0 dihedrals\n0 impropers\n\n")
        fh.write("4 atom types\n1 bond types\n1 angle types\n0 dihedral types\n0 improper types\n\n")
        for lab in "xyz":
            fh.write(f"0.0 {L:.6f} {lab}lo {lab}hi\n")
        fh.write("\nMasses\n\n" + "".join(f"{t} {m}\n" for t, m in masses.items()))
        fh.write("\nAtoms\n\n" + "".join(f"{a} {m} {t} {q:.5f} {x:.6f} {y:.6f} {z:.6f}\n" for a, m, t, q, x, y, z in atoms))
        fh.write("\nBonds\n\n" + "".join(f"{b} 1 {i} {j}\n" for b, i, j in bonds))
        fh.write("\nAngles\n\n" + "".join(f"{a} 1 {i} {j} {k}\n" for a, i, j, k in angles))
    return {"box": L, "n_atoms": len(atoms)}
