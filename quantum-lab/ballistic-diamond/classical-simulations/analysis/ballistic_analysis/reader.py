"""Readers for LAMMPS output files produced by the sweep.

For V1 we only need the heavy ion's final position, which lives in
`final.data` (a LAMMPS data file written by `write_data`).  We deliberately
skip the trajectory dumps -- parsing the ~1.5 M-atom-per-frame collision
trajectories for 5400 ensembles is the slow path and is not needed for
depth distributions, which is the primary observable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

# Match the per-ensemble directory layout written by deploy-experiment.sh:
#   .../tersoff-sweep[-pb]/energy-<E>/temperature-<T>/angle-<A>-<ensemble>
_DIRNAME_RE = re.compile(
    r"energy-(?P<E>[0-9.]+)/temperature-(?P<T>[0-9.]+)/angle-(?P<A>[0-9.]+)-(?P<ens>\d+)"
)


def parse_dirname(path: Path) -> Optional[dict]:
    """Pull (E_keV, angle_deg, T_K, ensemble) out of a per-ensemble path.

    Returns None if the path doesn't match the expected layout.
    """
    s = str(path)
    m = _DIRNAME_RE.search(s)
    if not m:
        return None
    return {
        "E_keV": float(m["E"]),
        "angle_deg": float(m["A"]),
        "T_K": float(m["T"]),
        "ensemble": int(m["ens"]),
    }


def read_ion_from_data(
    data_path: Path,
    ion_type: int = 2,
) -> Optional[dict]:
    """Parse a LAMMPS data file and return the ion's final state.

    Atomic style is `atomic`, so the Atoms section has columns
    `id type x y z` (optionally followed by image flags `ix iy iz`).
    Velocities section has `id vx vy vz`.

    Returns dict with keys (id, x, y, z, vx, vy, vz, ke_eV) for the (unique)
    atom of type=ion_type.  Returns None if no such atom is present (e.g. the
    ion was destroyed/dropped from the simulation).
    """
    section = None
    ion_id: Optional[int] = None
    ion_xyz: Optional[np.ndarray] = None
    ion_vel: Optional[np.ndarray] = None
    ion_image: Optional[np.ndarray] = None  # image flags, if present
    mass_ion: Optional[float] = None

    with open(data_path) as f:
        for raw in f:
            line = raw.strip()
            # Section headers in LAMMPS data files appear on their own line
            # and may be followed by inline comments (e.g. "Atoms # atomic").
            head = line.split("#", 1)[0].strip()
            if head in (
                "Atoms", "Velocities", "Masses",
                "Bonds", "Angles", "Dihedrals", "Impropers",
                "PairIJ Coeffs", "Pair Coeffs",
            ):
                section = head.lower()
                continue
            if not line or line.startswith("#"):
                continue

            if section == "masses":
                parts = line.split()
                # "<type> <mass>" -- grab the ion mass for KE conversion
                if len(parts) >= 2 and int(parts[0]) == ion_type:
                    mass_ion = float(parts[1])

            elif section == "atoms":
                parts = line.split()
                # atomic style: id type x y z [ix iy iz]
                if len(parts) < 5:
                    continue
                try:
                    atype = int(parts[1])
                except ValueError:
                    continue
                if atype != ion_type:
                    continue
                ion_id = int(parts[0])
                ion_xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                if len(parts) >= 8:
                    ion_image = np.array(
                        [int(parts[5]), int(parts[6]), int(parts[7])]
                    )

            elif section == "velocities":
                if ion_id is None:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                if int(parts[0]) != ion_id:
                    continue
                ion_vel = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )

    if ion_xyz is None:
        return None

    # KE = 0.5 * m * v^2; metal units (m in amu, v in A/ps, KE in eV).
    #   1 amu * (A/ps)^2 = 1.0364269e-4 eV
    ke_eV = 0.0
    if ion_vel is not None and mass_ion is not None:
        ke_eV = 0.5 * mass_ion * float(ion_vel @ ion_vel) * 1.0364269e-4

    out = {
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
        out["ix"] = int(ion_image[0])
        out["iy"] = int(ion_image[1])
        out["iz"] = int(ion_image[2])
    return out


def read_box_from_data(data_path: Path) -> dict:
    """Return the box bounds {xlo,xhi,ylo,yhi,zlo,zhi} from a LAMMPS data file."""
    box = {}
    with open(data_path) as f:
        for raw in f:
            line = raw.strip()
            if line.endswith("xlo xhi"):
                lo, hi = line.split()[:2]
                box["xlo"], box["xhi"] = float(lo), float(hi)
            elif line.endswith("ylo yhi"):
                lo, hi = line.split()[:2]
                box["ylo"], box["yhi"] = float(lo), float(hi)
            elif line.endswith("zlo zhi"):
                lo, hi = line.split()[:2]
                box["zlo"], box["zhi"] = float(lo), float(hi)
            elif "Atoms" in line:
                # box block is always before Atoms; bail once we hit it
                break
    return box
