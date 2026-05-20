"""Walk a sweep directory tree, extract per-ensemble observables, and pack
the result into an HDF5 file with one row per ensemble member.

The on-disk layout produced by deploy-experiment.sh is:
    <sweep_root>/energy-<E>/temperature-<T>/angle-<A>-<ensemble>/final.data

Output HDF5 layout (single file, queryable with h5py or pandas):
    /summary
        species         (S2)    "sn" or "pb"
        E_keV           (f8)
        angle_deg       (f8)
        T_K             (f8)
        ensemble        (i4)
        x, y, z         (f8)    ion final position [A]
        depth           (f8)    distance below the original top surface [A]
        vx, vy, vz      (f8)    ion final velocity [A/ps]
        ke_eV           (f8)    ion final kinetic energy [eV]
        ok              (i1)    1 if final.data found & ion present, else 0
    /attrs
        sweep_roots, species_list, n_total, n_ok
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ballistic_analysis.reader import parse_dirname, read_ion_from_data, read_box_from_data


@dataclass
class EnsembleRecord:
    species: str
    E_keV: float
    angle_deg: float
    T_K: float
    ensemble: int
    x: float
    y: float
    z: float
    depth: float
    vx: float
    vy: float
    vz: float
    ke_eV: float
    ok: int


def _empty_record(species: str, params: dict) -> EnsembleRecord:
    nan = float("nan")
    return EnsembleRecord(
        species=species,
        E_keV=params["E_keV"], angle_deg=params["angle_deg"],
        T_K=params["T_K"], ensemble=params["ensemble"],
        x=nan, y=nan, z=nan, depth=nan,
        vx=nan, vy=nan, vz=nan, ke_eV=nan, ok=0,
    )


def _process_one(args: tuple[str, Path]) -> Optional[EnsembleRecord]:
    species, ens_dir = args
    params = parse_dirname(ens_dir)
    if params is None:
        return None

    data_path = ens_dir / "final.data"
    if not data_path.is_file():
        return _empty_record(species, params)

    ion = read_ion_from_data(data_path)
    if ion is None:
        return _empty_record(species, params)

    box = read_box_from_data(data_path)
    # The slab originally extends from z=10*a to z=470*a (lattice units),
    # so the top surface sits at zhi - 30*a roughly.  For the depth metric
    # we use (top_of_slab - z_ion), where top_of_slab is approximated as
    # zhi - 30 A (30 A vacuum gap above the slab).  Adjust if you change
    # the slab geometry.
    top_of_slab = box.get("zhi", float("nan")) - 30.0

    return EnsembleRecord(
        species=species,
        E_keV=params["E_keV"], angle_deg=params["angle_deg"],
        T_K=params["T_K"], ensemble=params["ensemble"],
        x=ion["x"], y=ion["y"], z=ion["z"],
        depth=top_of_slab - ion["z"],
        vx=ion["vx"], vy=ion["vy"], vz=ion["vz"],
        ke_eV=ion["ke_eV"], ok=1,
    )


def walk_sweep_tree(sweep_root: Path, species: str) -> list[Path]:
    """Return every per-ensemble directory under sweep_root."""
    sweep_root = Path(sweep_root)
    if not sweep_root.is_dir():
        return []
    out = []
    for energy_dir in sweep_root.glob("energy-*"):
        for temp_dir in energy_dir.glob("temperature-*"):
            for ens_dir in temp_dir.glob("angle-*"):
                if ens_dir.is_dir() and parse_dirname(ens_dir) is not None:
                    out.append(ens_dir)
    return out


def build_summary(
    sweep_roots: dict[str, Path],
    n_workers: int = 1,
) -> list[EnsembleRecord]:
    """Build the full set of records by walking each (species, root) pair.

    sweep_roots is a dict like {"sn": Path(.../tersoff-sweep), "pb": Path(...)}.
    Set n_workers > 1 to parse final.data files in parallel.
    """
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
            for r in ex.map(_process_one, work, chunksize=32):
                if r is not None:
                    records.append(r)
    return records


def save_summary(records: Iterable[EnsembleRecord], out_path: Path) -> None:
    import h5py  # deferred so the package imports without h5py installed

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(records)
    n = len(records)
    if n == 0:
        raise ValueError("No records to save.")

    # Pack into a structured array for compact storage + easy round-tripping.
    cols = list(asdict(records[0]).keys())
    species_max_len = max(len(r.species) for r in records)
    dtype = []
    for c in cols:
        if c == "species":
            dtype.append((c, f"S{species_max_len}"))
        elif c in ("ensemble", "ok"):
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
        f.attrs["species_list"] = np.array(
            sorted({r.species for r in records}), dtype="S2"
        )


def load_summary(path: Path) -> np.ndarray:
    """Return the structured summary array (and decode species back to str)."""
    import h5py  # deferred

    with h5py.File(path, "r") as f:
        arr = f["summary"][...]
    # Re-cast species to a regular unicode column for ergonomic use downstream.
    species_str = np.array([s.decode() for s in arr["species"]])
    out_dtype = [
        (name, ("U8" if name == "species" else arr.dtype[name]))
        for name in arr.dtype.names
    ]
    out = np.zeros(arr.shape, dtype=out_dtype)
    for name in arr.dtype.names:
        out[name] = species_str if name == "species" else arr[name]
    return out
