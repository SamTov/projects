"""Utilities shared by per-experiment ``run.py`` drivers.

All experiments follow the same recipe:

1. Read ``config.yaml`` and the run index (--seed / --array-index) from CLI.
2. Map the index to a concrete parameter cell (temperature, bond count, ...).
3. Render a LAMMPS input file from a jinja2 template in ``lammps_templates/``.
4. Shell out to ``lmp`` (or whatever ``--lmp`` points to).
5. Load the resulting velocity dump.
6. Call ``rmt_dynamics`` to build C, compute the spectrum, and write
   per-run artifacts (``C.npz``, ``eigvals.npy``, ``row.csv``, ``meta.json``).

Experiment-specific analysis (figures, cross-run CSVs) happens in each
experiment's own ``analysis.py`` module, invoked after all array tasks
complete.

These helpers are intentionally small and engine-agnostic: swapping LAMMPS
for another MD engine would only require a new template + a new
``run_md`` implementation.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "lammps_templates"


@dataclass
class RunContext:
    """Everything one array-task needs to know about itself.

    `cell` is the concrete parameter assignment for this task (e.g.
    {"temperature": 100.0, "seed": 3, "n_atoms": 512}). `output_dir` is
    the per-task directory where trajectory, C, CSV row, and metadata
    are written. `template_name` is the file under lammps_templates/.
    """

    cell: dict[str, Any]
    output_dir: Path
    template_name: str
    lmp_binary: str = "lmp"
    lmp_flags: tuple[str, ...] = ()
    dry_run: bool = False
    skip_md: bool = False
    extra_context: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def enumerate_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the ``sweep`` block of a config into an ordered list of cells.

    The config's ``sweep`` entry is a dict of lists. The cartesian product is
    taken in the listed order, with the last key varying fastest. Each element
    of the returned list is a dict whose keys are the sweep's keys.

    Callers then merge this with the config's ``fixed`` block for rendering.
    """
    sweep = config.get("sweep", {})
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [list(sweep[k]) for k in keys]
    cells: list[dict[str, Any]] = []

    def _rec(idx: int, acc: dict[str, Any]) -> None:
        if idx == len(keys):
            cells.append(dict(acc))
            return
        for v in values[idx]:
            acc[keys[idx]] = v
            _rec(idx + 1, acc)

    _rec(0, {})
    return cells


def cell_tag(cell: dict[str, Any]) -> str:
    """Filesystem-safe identifier describing a single cell."""
    parts = []
    for k, v in cell.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts) if parts else "default"


def render_template(template_name: str, context: dict[str, Any]) -> str:
    """Render a jinja2 template from ``lammps_templates/`` to a string."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(**context)


def run_md(input_file: Path, log_file: Path, ctx: RunContext) -> None:
    """Invoke LAMMPS on ``input_file``, capturing stdout/stderr to ``log_file``.

    Honours ``ctx.dry_run`` (skip execution) and ``ctx.skip_md`` (don't run
    MD this session; assume a prior dump exists). Non-zero exit codes raise.
    """
    if ctx.dry_run or ctx.skip_md:
        return
    # lmp_binary may include an MPI wrapper, e.g. "mpirun -n 8 lmp".
    cmd_prefix = shlex.split(ctx.lmp_binary)
    cmd = [*cmd_prefix, *ctx.lmp_flags, "-in", str(input_file)]
    with open(log_file, "w") as fh:
        proc = subprocess.run(
            cmd, cwd=str(ctx.output_dir), stdout=fh, stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LAMMPS exited with code {proc.returncode}; see {log_file}"
        )


@dataclass
class DumpData:
    """In-memory representation of a LAMMPS custom dump.

    velocities and positions are None when their columns aren't present.
    Position columns are picked in this order: xu/yu/zu (unwrapped),
    x/y/z (wrapped), xs/ys/zs (fractional — converted using box).
    """

    velocities: np.ndarray | None
    positions: np.ndarray | None
    atom_types: np.ndarray
    box: np.ndarray  # (3,) lengths


def parse_lammps_dump_custom(path: Path) -> DumpData:
    """Parse a LAMMPS ``custom`` dump.

    Reads any combination of velocity (`vx vy vz`) and position
    (`xu yu zu` preferred, then `x y z`) columns. Returns a `DumpData`
    with whichever are present. Atom IDs are remapped to 0-based and frames
    are stored sorted by id.
    """
    vel_frames: list[np.ndarray] = []
    pos_frames: list[np.ndarray] = []
    types: np.ndarray | None = None
    box_last: np.ndarray | None = None

    have_vel: bool | None = None
    have_pos_unwrapped: bool | None = None
    have_pos_wrapped: bool | None = None

    with open(path, "r") as fh:
        line = fh.readline()
        n_atoms = 0
        while line:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(fh.readline())
            elif line.startswith("ITEM: BOX BOUNDS"):
                box_lengths = np.empty(3, dtype=np.float64)
                for d in range(3):
                    parts = fh.readline().split()
                    box_lengths[d] = float(parts[1]) - float(parts[0])
                box_last = box_lengths
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                if have_vel is None:
                    have_vel = all(c in cols for c in ("vx", "vy", "vz"))
                    have_pos_unwrapped = all(c in cols for c in ("xu", "yu", "zu"))
                    have_pos_wrapped = (not have_pos_unwrapped) and all(
                        c in cols for c in ("x", "y", "z")
                    )
                idx_id = cols.index("id")
                idx_type = cols.index("type")
                vel = np.empty((n_atoms, 3), dtype=np.float64) if have_vel else None
                pos = (
                    np.empty((n_atoms, 3), dtype=np.float64)
                    if (have_pos_unwrapped or have_pos_wrapped)
                    else None
                )
                frame_types = np.empty(n_atoms, dtype=np.int64)
                vi = [cols.index(c) for c in ("vx", "vy", "vz")] if have_vel else None
                if have_pos_unwrapped:
                    pi = [cols.index(c) for c in ("xu", "yu", "zu")]
                elif have_pos_wrapped:
                    pi = [cols.index(c) for c in ("x", "y", "z")]
                else:
                    pi = None
                for _ in range(n_atoms):
                    fields = fh.readline().split()
                    aid = int(fields[idx_id]) - 1
                    frame_types[aid] = int(fields[idx_type])
                    if vel is not None and vi is not None:
                        vel[aid, 0] = float(fields[vi[0]])
                        vel[aid, 1] = float(fields[vi[1]])
                        vel[aid, 2] = float(fields[vi[2]])
                    if pos is not None and pi is not None:
                        pos[aid, 0] = float(fields[pi[0]])
                        pos[aid, 1] = float(fields[pi[1]])
                        pos[aid, 2] = float(fields[pi[2]])
                if vel is not None:
                    vel_frames.append(vel)
                if pos is not None:
                    pos_frames.append(pos)
                if types is None:
                    types = frame_types
            line = fh.readline()

    if types is None:
        raise RuntimeError(f"No atom records parsed from {path}")
    if box_last is None:
        box_last = np.zeros(3, dtype=np.float64)

    velocities = np.stack(vel_frames, axis=0) if vel_frames else None
    positions = np.stack(pos_frames, axis=0) if pos_frames else None
    return DumpData(velocities=velocities, positions=positions,
                    atom_types=types, box=box_last)


def write_row(path: Path, row: dict[str, Any]) -> None:
    """Append a single CSV row; writes header if the file doesn't exist yet."""
    import csv

    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_meta(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)


def pick_cell(cells: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Clamp-and-index helper for --array-index. Raises on out-of-range."""
    if index < 0 or index >= len(cells):
        raise IndexError(
            f"array index {index} out of range (sweep has {len(cells)} cells)"
        )
    return cells[index]


def resolve_run_ctx(
    config: dict[str, Any],
    cell: dict[str, Any],
    output_root: Path,
    template_name: str,
    lmp_binary: str = "lmp",
    lmp_flags: Iterable[str] = (),
    dry_run: bool = False,
    skip_md: bool = False,
    extra_context: dict[str, Any] | None = None,
) -> RunContext:
    """Pack per-run paths and flags into a RunContext."""
    tag = cell_tag(cell)
    output_dir = output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        cell=cell,
        output_dir=output_dir,
        template_name=template_name,
        lmp_binary=lmp_binary,
        lmp_flags=tuple(lmp_flags),
        dry_run=dry_run,
        skip_md=skip_md,
        extra_context=dict(extra_context or {}),
    )


def add_common_args(parser) -> None:
    """Register flags shared by every experiment's ``run.py``."""
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML config describing the sweep and fixed parameters.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Where to write trajectories, C, and summary tables.",
    )
    parser.add_argument(
        "--array-index", type=int, default=None,
        help="Which cell of the sweep to run (0-based). If omitted, "
             "the driver prints the sweep size and exits.",
    )
    parser.add_argument(
        "--lmp", type=str, default="lmp",
        help="LAMMPS binary to invoke (default: 'lmp' in PATH).",
    )
    parser.add_argument(
        "--mpi", type=int, default=None,
        help="If set, prefixes the LAMMPS command with `mpirun -n <N>`.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Render inputs and enumerate work but do not run LAMMPS.",
    )
    parser.add_argument(
        "--skip-md", action="store_true",
        help="Skip the MD step; reuse an existing dump under output-dir.",
    )


def lmp_flags_from_args(args) -> tuple[str, ...]:
    """Translate `--mpi` into the right invocation flags."""
    if args.mpi is not None and args.mpi > 1:
        return ()  # MPI is handled by wrapping the binary elsewhere.
    return ()


def lmp_binary_with_mpi(args) -> str:
    if args.mpi is not None and args.mpi > 1:
        return f"mpirun -n {args.mpi} {args.lmp}"
    return args.lmp


def print_sweep(cells: list[dict[str, Any]]) -> None:
    """Human-readable listing of the sweep with its array indices."""
    print(f"Sweep size: {len(cells)}")
    for i, cell in enumerate(cells):
        print(f"  [{i}] {cell}")
