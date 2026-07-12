"""Readers for LAMMPS output files produced by the sweep.

Primary entry points:
  read_final_state()     -- one streaming pass over final.data: ion record,
                            box bounds, and a data-driven estimate of the slab
                            surface/bottom (from the carbon z-histogram).
                            The surface must be measured, not assumed: the
                            vacuum gap is 30 *lattice units* (~151 A), and a
                            hardcoded offset once cost us a 121 A depth bias.
  read_ion_trajectory()  -- parse the ion-only dump (collision-ion /
                            anneal-ion .lammpstraj); used to classify ions
                            missing from final.data (transmitted/reflected).
  parse_dirname()        -- recover (E, angle, T, ensemble) from a path.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

# Match the per-ensemble directory layout written by the sweep:
#   .../orient-<O>/energy-<E>/temperature-<T>/angle-<A>-<ensemble>
# The orient level is optional for backward compatibility with pre-orientation
# runs and the flat test layout; absent means the original 110 geometry.
_DIRNAME_RE = re.compile(
    r"(?:orient-(?P<O>\d+)/)?"
    r"energy-(?P<E>[0-9.]+)/temperature-(?P<T>[0-9.]+)/angle-(?P<A>[0-9.]+)-(?P<ens>\d+)"
)


def parse_dirname(path: Path) -> Optional[dict]:
    """Pull (orientation, E_keV, angle_deg, T_K, ensemble) from a path."""
    m = _DIRNAME_RE.search(str(path))
    if not m:
        return None
    return {
        "orientation": int(m["O"]) if m["O"] else 110,
        "E_keV": float(m["E"]),
        "angle_deg": float(m["A"]),
        "T_K": float(m["T"]),
        "ensemble": int(m["ens"]),
    }


_SECTION_HEADS = {
    "Atoms", "Velocities", "Masses",
    "Bonds", "Angles", "Dihedrals", "Impropers",
    "PairIJ Coeffs", "Pair Coeffs",
}


def read_final_state(
    data_path: Path,
    ion_type: int = 2,
    bin_A: float = 1.0,
) -> Optional[dict]:
    """Parse a LAMMPS data file (atomic style) in one streaming pass.

    Returns a dict:
      ion        : dict(id,x,y,z,vx,vy,vz,ke_eV[,ix,iy,iz]) or None if absent
      box        : {xlo,xhi,ylo,yhi,zlo,zhi}
      z_surface  : upper edge of the highest z-bin with >= half the bulk
                   carbon count (the top surface, adatoms excluded)
      z_bottom   : mirror-image estimate of the slab bottom
      n_carbon   : carbon atom count

    Returns None if the file is unreadable/empty.
    """
    box: dict = {}
    section = None
    ion_id: Optional[int] = None
    ion_xyz = ion_vel = ion_image = None
    mass_ion: Optional[float] = None
    n_carbon = 0
    hist: Optional[np.ndarray] = None
    zlo = zhi = None
    nbins = 0

    try:
        f = open(data_path)
    except OSError:
        return None

    with f:
        for raw in f:
            line = raw.strip()
            if zhi is None:
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
                    zlo, zhi = float(lo), float(hi)
                    box["zlo"], box["zhi"] = zlo, zhi
                    nbins = max(1, int(np.ceil((zhi - zlo) / bin_A)))
                    hist = np.zeros(nbins, dtype=np.int64)
                    continue

            head = line.split("#", 1)[0].strip()
            if head in _SECTION_HEADS:
                section = head.lower()
                continue
            if not line or line.startswith("#"):
                continue

            if section == "masses":
                parts = line.split()
                if len(parts) >= 2 and int(parts[0]) == ion_type:
                    mass_ion = float(parts[1])

            elif section == "atoms":
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    atype = int(parts[1])
                    z = float(parts[4])
                except ValueError:
                    continue
                if atype == ion_type:
                    ion_id = int(parts[0])
                    ion_xyz = np.array(
                        [float(parts[2]), float(parts[3]), z]
                    )
                    if len(parts) >= 8:
                        ion_image = np.array(
                            [int(parts[5]), int(parts[6]), int(parts[7])]
                        )
                else:
                    n_carbon += 1
                    if hist is not None:
                        idx = int((z - zlo) / bin_A)
                        if 0 <= idx < nbins:
                            hist[idx] += 1

            elif section == "velocities":
                if ion_id is None:
                    continue
                parts = line.split()
                if len(parts) >= 4 and int(parts[0]) == ion_id:
                    ion_vel = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )

    if not box or hist is None or n_carbon == 0:
        return None

    # Surface/bottom: median count of occupied bins ~ bulk layer density;
    # extreme bins with >= half that are real material, sparser ones are
    # adatoms / sputter debris.
    occupied = hist[hist > 0]
    bulk = float(np.median(occupied))
    material = np.nonzero(hist >= 0.5 * bulk)[0]
    z_surface = zlo + (material[-1] + 1) * bin_A
    z_bottom = zlo + material[0] * bin_A

    ion = None
    if ion_xyz is not None:
        ke_eV = 0.0
        if ion_vel is not None and mass_ion is not None:
            # metal units: 1 amu (A/ps)^2 = 1.0364269e-4 eV
            ke_eV = 0.5 * mass_ion * float(ion_vel @ ion_vel) * 1.0364269e-4
        ion = {
            "id": int(ion_id),
            "x": float(ion_xyz[0]),
            "y": float(ion_xyz[1]),
            "z": float(ion_xyz[2]),
            "vx": float(ion_vel[0]) if ion_vel is not None else float("nan"),
            "vy": float(ion_vel[1]) if ion_vel is not None else float("nan"),
            "vz": float(ion_vel[2]) if ion_vel is not None else float("nan"),
            "ke_eV": ke_eV,
        }
        if ion_image is not None:
            ion["ix"], ion["iy"], ion["iz"] = (int(v) for v in ion_image)

    return {
        "ion": ion,
        "box": box,
        "z_surface": float(z_surface),
        "z_bottom": float(z_bottom),
        "n_carbon": n_carbon,
    }


def read_ion_trajectory(dump_path: Path) -> Optional[dict]:
    """Parse an ion-only LAMMPS custom dump (one atom per frame).

    Returns {"step": (F,), "pos": (F,3), "vel": (F,3)} with only the frames
    where the ion was present, or None if the file is missing/has no frames.
    """
    steps: list[int] = []
    pos: list[list[float]] = []
    vel: list[list[float]] = []

    try:
        f = open(dump_path)
    except OSError:
        return None

    with f:
        lines = iter(f)
        for line in lines:
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            try:
                step = int(next(lines).strip())
                assert next(lines).startswith("ITEM: NUMBER OF ATOMS")
                natoms = int(next(lines).strip())
                # Skip box-bounds block (header + 3 lines)
                assert next(lines).startswith("ITEM: BOX BOUNDS")
                for _ in range(3):
                    next(lines)
                header = next(lines).strip()
                assert header.startswith("ITEM: ATOMS")
                cols = header.split()[2:]
                for _ in range(natoms):
                    parts = next(lines).split()
                    row = dict(zip(cols, parts))
                    steps.append(step)
                    pos.append([float(row["x"]), float(row["y"]), float(row["z"])])
                    vel.append([
                        float(row.get("vx", "nan")),
                        float(row.get("vy", "nan")),
                        float(row.get("vz", "nan")),
                    ])
            except (StopIteration, AssertionError, KeyError, ValueError):
                break  # truncated file (job died mid-write) -- keep what we have

    if not steps:
        return None
    return {
        "step": np.array(steps),
        "pos": np.array(pos),
        "vel": np.array(vel),
    }


def read_all_atoms(data_path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (types, positions[N,3]) for every atom in a LAMMPS data file.

    Used by the damage module, which needs the full carbon set.  The Atoms
    section of a write_data atomic-style file has uniform columns
    (id type x y z [ix iy iz]), so the block parses in one numpy call.
    """
    lines: list[str] = []
    in_atoms = False
    try:
        with open(data_path) as f:
            for raw in f:
                stripped = raw.strip()
                head = stripped.split("#", 1)[0].strip()
                if head == "Atoms":
                    in_atoms = True
                    continue
                if in_atoms:
                    if head in _SECTION_HEADS:
                        break
                    if stripped and not stripped.startswith("#"):
                        lines.append(stripped)
    except OSError:
        return None
    if not lines:
        return None
    arr = np.loadtxt(lines, ndmin=2)
    return arr[:, 1].astype(int), arr[:, 2:5]


# ---------------------------------------------------------------------------
# Back-compat helpers (older notebooks / scripts)
# ---------------------------------------------------------------------------

def read_ion_from_data(data_path: Path, ion_type: int = 2) -> Optional[dict]:
    """Ion record only (legacy API)."""
    state = read_final_state(data_path, ion_type=ion_type)
    return None if state is None else state["ion"]


def read_box_from_data(data_path: Path) -> dict:
    """Box bounds only (legacy API)."""
    state = read_final_state(data_path)
    return {} if state is None else state["box"]
