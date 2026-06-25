"""Trajectory and correlation-matrix persistence.

`load_velocities` is MDAnalysis-backed and imported lazily so the bulk of
the package (and its tests) work without the `[io]` extra installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load_velocities", "save_C", "load_C"]


def load_velocities(
    topology: str | os.PathLike,
    trajectory: str | os.PathLike,
    selection: str = "all",
    stride: int = 1,
) -> tuple[np.ndarray, float]:
    """Load atomic velocities from an MDAnalysis-readable trajectory.

    Returns
    -------
    velocities : (n_frames_sampled, N_selected, 3) float64
    dt_sampled : float, time between sampled frames (trajectory dt × stride).

    Notes
    -----
    Requires the `[io]` extra (`pip install -e ".[io]"`). MDAnalysis is
    imported lazily so code that doesn't touch trajectories can run without
    it. Errors from MDAnalysis propagate unchanged.
    """
    try:
        import MDAnalysis as mda  # noqa: F401  (imported lazily)
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "load_velocities requires MDAnalysis; install with '.[io]'"
        ) from exc

    u = mda.Universe(str(topology), str(trajectory))
    atoms = u.select_atoms(selection)
    n_frames_total = len(u.trajectory)
    frame_indices = list(range(0, n_frames_total, int(stride)))
    if not frame_indices:
        raise ValueError("stride produced an empty frame list")

    n_selected = atoms.n_atoms
    velocities = np.empty((len(frame_indices), n_selected, 3), dtype=np.float64)
    dt_base = float(u.trajectory.dt)
    for out_idx, frame_idx in enumerate(frame_indices):
        u.trajectory[frame_idx]
        if atoms.velocities is None:
            raise RuntimeError(
                f"frame {frame_idx} has no velocities; check the trajectory format"
            )
        velocities[out_idx] = atoms.velocities
    return velocities, dt_base * int(stride)


def save_C(
    path: str | os.PathLike,
    C: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a correlation matrix as `path` (`.npz`) with a `.json` sidecar.

    The `.npz` file contains a single array `C`. Metadata is written as JSON
    next to the npz (`path` with suffix replaced by `.json`). An empty or
    omitted metadata dict still writes an empty JSON object for provenance.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"save_C expects a .npz path; got '{path}'")
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square 2-D; got shape {C.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, C=C)

    sidecar = path.with_suffix(".json")
    payload = {} if metadata is None else dict(metadata)
    payload.setdefault("shape", list(C.shape))
    payload.setdefault("dtype", str(C.dtype))
    with sidecar.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_C(path: str | os.PathLike) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a correlation matrix saved by `save_C`.

    Returns
    -------
    C : (N, N) float64 ndarray
    metadata : dict parsed from the JSON sidecar (empty if absent).
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"load_C expects a .npz path; got '{path}'")
    with np.load(path) as data:
        if "C" not in data.files:
            raise KeyError(f"'C' not found in {path} (keys: {list(data.files)})")
        C = np.asarray(data["C"], dtype=np.float64)
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        with sidecar.open() as fh:
            metadata = json.load(fh)
    else:
        metadata = {}
    return C, metadata
