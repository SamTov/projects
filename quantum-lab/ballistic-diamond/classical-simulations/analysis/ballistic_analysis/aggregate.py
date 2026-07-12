"""Walk a sweep directory tree, extract per-ensemble observables, and pack
the result into an HDF5 file with one row per ensemble member.

On-disk layout per ensemble (written by simulate.lmp into scratch):
    <sweep_root>/energy-<E>/temperature-<T>/angle-<A>-<ens>/
        params.json                 (azimuth + provenance)
        collision-ion.lammpstraj    (ion-only cascade trajectory)
        anneal-ion.lammpstraj       (ion-only quench trajectory)
        final.data                  (full-slab snapshot)

Output HDF5 (/summary, one structured row per ensemble):
    species, E_keV, angle_deg, azimuth_deg, T_K, ensemble,
    x, y, z            ion final position [A]
    depth              z_surface - z (surface measured from carbon density)
    surface_z          measured top-surface height [A]
    vx, vy, vz, ke_eV  ion final velocity / KE
    status             implanted | transmitted | reflected | lost | no_data
    ok                 1 iff status == implanted

Status logic when the ion is absent from final.data (open z boundary lets
it leave): the last frame of the ion trajectory decides -- below the slab
bottom = transmitted (punch-through), above the surface = reflected,
otherwise lost (indeterminate, e.g. NaN'd).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ballistic_analysis.reader import (
    parse_dirname,
    read_final_state,
    read_ion_trajectory,
)

_STATUS = ("implanted", "transmitted", "reflected", "lost", "no_data")


@dataclass
class EnsembleRecord:
    species: str
    orientation: int
    E_keV: float
    angle_deg: float
    azimuth_deg: float
    T_K: float
    ensemble: int
    x: float
    y: float
    z: float
    depth: float
    surface_z: float
    vx: float
    vy: float
    vz: float
    ke_eV: float
    status: str
    ok: int


def _base_record(species: str, params: dict, azimuth: float) -> EnsembleRecord:
    nan = float("nan")
    return EnsembleRecord(
        species=species,
        orientation=params["orientation"],
        E_keV=params["E_keV"], angle_deg=params["angle_deg"],
        azimuth_deg=azimuth,
        T_K=params["T_K"], ensemble=params["ensemble"],
        x=nan, y=nan, z=nan, depth=nan, surface_z=nan,
        vx=nan, vy=nan, vz=nan, ke_eV=nan,
        status="no_data", ok=0,
    )


def _read_azimuth(ens_dir: Path) -> float:
    try:
        with open(ens_dir / "params.json") as f:
            return float(json.load(f)["azimuth_deg"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return float("nan")


def _process_one(args: tuple[str, Path]) -> Optional[EnsembleRecord]:
    species, ens_dir = args
    params = parse_dirname(ens_dir)
    if params is None:
        return None

    rec = _base_record(species, params, _read_azimuth(ens_dir))

    state = read_final_state(ens_dir / "final.data")
    if state is None:
        return rec  # no_data

    rec.surface_z = state["z_surface"]
    ion = state["ion"]

    if ion is not None:
        rec.x, rec.y, rec.z = ion["x"], ion["y"], ion["z"]
        rec.vx, rec.vy, rec.vz = ion["vx"], ion["vy"], ion["vz"]
        rec.ke_eV = ion["ke_eV"]
        rec.depth = state["z_surface"] - ion["z"]
        rec.status = "implanted"
        rec.ok = 1
        return rec

    # Ion left the box: classify from its last known trajectory point.
    rec.status = "lost"
    traj = read_ion_trajectory(ens_dir / "collision-ion.lammpstraj")
    if traj is not None:
        z_last = float(traj["pos"][-1, 2])
        if z_last < state["z_bottom"] + 20.0:
            rec.status = "transmitted"
        elif z_last > state["z_surface"] - 20.0:
            rec.status = "reflected"
    return rec


def walk_sweep_tree(sweep_root: Path, species: str) -> list[Path]:
    """Return every per-ensemble directory under sweep_root."""
    sweep_root = Path(sweep_root)
    if not sweep_root.is_dir():
        return []
    out = []
    orient_dirs = sorted(sweep_root.glob("orient-*")) or [sweep_root]
    for level in orient_dirs:
        for energy_dir in level.glob("energy-*"):
            for temp_dir in energy_dir.glob("temperature-*"):
                for ens_dir in temp_dir.glob("angle-*"):
                    if ens_dir.is_dir() and parse_dirname(ens_dir) is not None:
                        out.append(ens_dir)
    return out


def build_summary(
    sweep_roots: dict[str, Path],
    n_workers: int = 1,
) -> list[EnsembleRecord]:
    """Build records for each (species, root); n_workers > 1 parallelises."""
    work: list[tuple[str, Path]] = []
    for species, root in sweep_roots.items():
        for ens_dir in walk_sweep_tree(root, species):
            work.append((species, ens_dir))

    records: list[EnsembleRecord] = []
    if n_workers <= 1:
        for item in work:
            r = _process_one(item)
            if r is not None:
                records.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for r in ex.map(_process_one, work, chunksize=16):
                if r is not None:
                    records.append(r)
    return records


_STR_COLS = ("species", "status")
_INT_COLS = ("orientation", "ensemble", "ok")


def save_summary(records: Iterable[EnsembleRecord], out_path: Path) -> None:
    import h5py  # deferred so the package imports without h5py installed

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(records)
    n = len(records)
    if n == 0:
        raise ValueError("No records to save.")

    cols = list(asdict(records[0]).keys())
    dtype = []
    for c in cols:
        if c in _STR_COLS:
            width = max(len(getattr(r, c)) for r in records)
            dtype.append((c, f"S{max(width, 1)}"))
        elif c in _INT_COLS:
            dtype.append((c, "i4"))
        else:
            dtype.append((c, "f8"))
    arr = np.zeros(n, dtype=dtype)
    for i, r in enumerate(records):
        for c in cols:
            v = getattr(r, c)
            arr[c][i] = v.encode() if isinstance(v, str) else v

    with h5py.File(out_path, "w") as f:
        f.create_dataset("summary", data=arr, compression="gzip")
        f.attrs["n_total"] = n
        f.attrs["n_ok"] = int(arr["ok"].sum())
        for status in _STATUS:
            f.attrs[f"n_{status}"] = int(
                (arr["status"] == status.encode()).sum()
            )


def load_summary(path: Path) -> np.ndarray:
    """Return the structured summary array with string columns decoded."""
    import h5py  # deferred

    with h5py.File(path, "r") as f:
        arr = f["summary"][...]

    out_dtype = []
    for name in arr.dtype.names:
        if arr.dtype[name].kind == "S":
            out_dtype.append((name, "U16"))
        else:
            out_dtype.append((name, arr.dtype[name]))
    out = np.zeros(arr.shape, dtype=out_dtype)
    for name in arr.dtype.names:
        if arr.dtype[name].kind == "S":
            out[name] = np.array([s.decode() for s in arr[name]])
        else:
            out[name] = arr[name]
    return out
