"""Build a LAMMPS data file for TIP4P/2005 water.

Places `n_water` molecules on a cubic lattice with random orientations at
approximately liquid density, writes out an atom_style=full data file with
O-H bonds and H-O-H angles so LAMMPS + SHAKE can keep the geometry rigid.

The massless M site is synthesised by the ``lj/cut/tip4p/long`` pair style
at run time and is NOT present in the data file.

`build_ice_ih_data` builds a proton-disordered ice Ih crystal that obeys
the Bernal-Fowler rules: the oxygen lattice is generated from the
hexagonal unit cell, the hydrogen-bond network is oriented along a random
Euler circuit of the four-coordinated oxygen graph (which gives every
oxygen exactly two donated and two accepted bonds), and directed cycles are
then reversed at random to drive the net dipole towards zero.

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
Q_O = -1.1128   # carried on the O atom; the pair style moves it to the M site
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


def _write_water_data(path: Path, coords: np.ndarray, box: np.ndarray, title: str) -> None:
    """Write (n_water, 3, 3) O/H/H coordinates as an atom_style full data file."""
    n_water = coords.shape[0]
    with open(path, "w") as fh:
        fh.write(f"LAMMPS data file for TIP4P/2005 water ({title})\n\n")
        fh.write(f"{3 * n_water} atoms\n{2 * n_water} bonds\n{n_water} angles\n")
        fh.write("0 dihedrals\n0 impropers\n\n")
        fh.write("2 atom types\n1 bond types\n1 angle types\n")
        fh.write("0 dihedral types\n0 improper types\n\n")
        for d, lab in enumerate("xyz"):
            fh.write(f"0.0 {box[d]:.6f} {lab}lo {lab}hi\n")
        fh.write("\nMasses\n\n1 15.9994\n2 1.00784\n\nAtoms\n\n")
        for m in range(n_water):
            for a, (t, q) in enumerate(((1, Q_O), (2, Q_H), (2, Q_H))):
                x, y, z = coords[m, a]
                fh.write(f"{3 * m + a + 1} {m + 1} {t} {q:.4f} {x:.6f} {y:.6f} {z:.6f}\n")
        fh.write("\nBonds\n\n")
        for m in range(n_water):
            fh.write(f"{2 * m + 1} 1 {3 * m + 1} {3 * m + 2}\n{2 * m + 2} 1 {3 * m + 1} {3 * m + 3}\n")
        fh.write("\nAngles\n\n")
        for m in range(n_water):
            fh.write(f"{m + 1} 1 {3 * m + 2} {3 * m + 1} {3 * m + 3}\n")


def ice_ih_oxygens(nx: int, ny: int, nz: int, a: float = 4.52, c: float = 7.36) -> tuple[np.ndarray, np.ndarray]:
    """Oxygen sites of ice Ih in an orthorhombic (a, sqrt(3) a, c) supercell.

    P6_3/mmc with O on the 4f sites (1/3, 2/3, z), (2/3, 1/3, z + 1/2),
    (2/3, 1/3, -z), (1/3, 2/3, 1/2 - z), z = 1/16; each orthorhombic cell
    holds 8 molecules. Returns (positions (8 nx ny nz, 3), box (3,)).
    """
    box = np.array([nx * a, ny * np.sqrt(3.0) * a, nz * c])
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0])
    a3 = np.array([0.0, 0.0, c])
    z = 1.0 / 16.0
    basis = np.array([[1 / 3, 2 / 3, z], [2 / 3, 1 / 3, z + 0.5],
                      [2 / 3, 1 / 3, -z], [1 / 3, 2 / 3, 0.5 - z]])
    pts = []
    span = 2 * (nx + ny) + 4
    for i in range(-span, span):
        for j in range(-span, span):
            for k in range(nz):
                for f in basis:
                    pts.append((i + f[0]) * a1 + (j + f[1]) * a2 + (k + f[2]) * a3)
    pts = np.mod(np.array(pts), box)
    key = np.round(pts / 1e-3).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    pts = pts[np.sort(keep)]
    # Fold sites that sit on the far box face onto the near one.
    pts = np.mod(pts + 1e-6, box) - 1e-6
    key = np.round(np.mod(pts, box) / 1e-2).astype(np.int64) % np.round(box / 1e-2).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    pts = np.mod(pts[np.sort(keep)], box)
    if len(pts) != 8 * nx * ny * nz:
        raise RuntimeError(f"expected {8 * nx * ny * nz} oxygens, built {len(pts)}")
    return pts, box


def _min_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    return d - box * np.round(d / box)


def ice_ih_hydrogen_bonds(o_pos: np.ndarray, box: np.ndarray, seed: int, max_sweeps: int = 20000):
    """Directed O->O donor edges obeying the ice rules with near-zero dipole.

    Returns an (n, 2) array: for each oxygen, the indices of the two
    oxygens it donates a hydrogen to.
    """
    rng = np.random.default_rng(seed)
    n = len(o_pos)
    d = _min_image(o_pos[None, :, :] - o_pos[:, None, :], box)
    r = np.linalg.norm(d, axis=-1)
    nbrs = [np.where((r[i] > 0.1) & (r[i] < 3.1))[0] for i in range(n)]
    if any(len(x) != 4 for x in nbrs):
        raise RuntimeError("oxygen lattice is not four-coordinated")
    edges = sorted({(min(i, j), max(i, j)) for i in range(n) for j in nbrs[i]})
    # Random Euler circuit (Hierholzer): orienting every edge along the
    # circuit gives in-degree = out-degree = 2 at each vertex.
    adj = {i: [] for i in range(n)}
    for e_id, (i, j) in enumerate(edges):
        adj[i].append((j, e_id))
        adj[j].append((i, e_id))
    for i in adj:
        rng.shuffle(adj[i])
    used = np.zeros(len(edges), dtype=bool)
    direction = {}
    stack = [(int(rng.integers(n)), None, None)]
    ptr = {i: 0 for i in range(n)}
    while stack:
        v, _, _ = stack[-1]
        while ptr[v] < len(adj[v]) and used[adj[v][ptr[v]][1]]:
            ptr[v] += 1
        if ptr[v] == len(adj[v]):
            stack.pop()
            continue
        w, e_id = adj[v][ptr[v]]
        used[e_id] = True
        direction[e_id] = (v, w)
        stack.append((w, v, e_id))
    out = {i: set() for i in range(n)}
    for v, w in direction.values():
        out[v].add(w)
    assert all(len(out[i]) == 2 for i in range(n))

    def dipole():
        m = np.zeros(3)
        for i in range(n):
            for j in out[i]:
                m += _min_image(o_pos[j] - o_pos[i], box)
        return m

    m = dipole()
    target = 0.02 * 2 * n * 2.75 / np.sqrt(n)  # well below a random orientation
    for _ in range(max_sweeps):
        if np.linalg.norm(m) < target:
            break
        # Follow donor edges from a random oxygen until a site repeats; the
        # loop found is a directed cycle whose reversal keeps the ice rules.
        start = int(rng.integers(n))
        path, seen = [start], {start: 0}
        while True:
            nxt = int(rng.choice(sorted(out[path[-1]])))
            if nxt in seen:
                cycle = path[seen[nxt]:] + [nxt]
                break
            seen[nxt] = len(path)
            path.append(nxt)
        dm = np.zeros(3)
        for v, w in zip(cycle[:-1], cycle[1:]):
            dm -= 2.0 * _min_image(o_pos[w] - o_pos[v], box)
        if np.linalg.norm(m + dm) < np.linalg.norm(m):
            for v, w in zip(cycle[:-1], cycle[1:]):
                out[v].remove(w)
                out[w].add(v)
            m = m + dm
    return np.array([sorted(out[i]) for i in range(n)]), float(np.linalg.norm(m))


def build_ice_ih_data(path: Path, nx: int, ny: int, nz: int, seed: int) -> dict:
    """Proton-disordered ice Ih data file; returns a small summary dict."""
    o_pos, box = ice_ih_oxygens(nx, ny, nz)
    donors, dip = ice_ih_hydrogen_bonds(o_pos, box, seed)
    half = np.deg2rad(HOH_ANGLE / 2.0)
    coords = np.empty((len(o_pos), 3, 3))
    for i, (j, k) in enumerate(donors):
        u1 = _min_image(o_pos[j] - o_pos[i], box)
        u2 = _min_image(o_pos[k] - o_pos[i], box)
        u1 /= np.linalg.norm(u1)
        u2 /= np.linalg.norm(u2)
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        perp = (u1 - u2) / np.linalg.norm(u1 - u2)
        coords[i, 0] = o_pos[i]
        coords[i, 1] = o_pos[i] + OH_BOND * (np.cos(half) * bis + np.sin(half) * perp)
        coords[i, 2] = o_pos[i] + OH_BOND * (np.cos(half) * bis - np.sin(half) * perp)
    coords = np.mod(coords, box)
    _write_water_data(path, coords, box, "proton-disordered ice Ih")
    return {"n_water": len(o_pos), "box": box.tolist(), "residual_dipole_A": dip}


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
