"""Experiment 3 — LJ phases (NPT scan). Computes ρ(T) and S(C, T)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    add_common_args, enumerate_cells, load_config, parse_lammps_dump_custom,
    pick_cell, print_sweep, render_template, resolve_run_ctx, run_md,
    write_meta, write_row,
)

from rmt_dynamics import (  # noqa: E402
    build_C, eigenvalues, estimate_tau_int, save_C, vn_entropy,
)


def _cubic_box_length(n_atoms: int, density_reduced: float, sigma: float) -> float:
    number_density_inv_vol = density_reduced / (sigma ** 3)
    return float((n_atoms / number_density_inv_vol) ** (1.0 / 3.0))


def _parse_density_from_log(log_path: Path) -> float:
    """Average the 'density' column in LAMMPS's thermo output over production.

    Returns NaN if the log doesn't contain a density column.
    """
    text = log_path.read_text(errors="ignore")
    # Thermo blocks start with a "Step" header. We take the LAST block (prod).
    blocks = re.findall(r"^Step.*?^Loop time", text, flags=re.MULTILINE | re.DOTALL)
    if not blocks:
        return float("nan")
    block = blocks[-1]
    header = block.splitlines()[0].split()
    if "Density" not in header:
        return float("nan")
    idx = header.index("Density")
    rows = block.splitlines()[1:-1]  # skip header + "Loop time"
    values: list[float] = []
    for row in rows:
        parts = row.split()
        if len(parts) > idx:
            try:
                values.append(float(parts[idx]))
            except ValueError:
                continue
    return float(np.mean(values)) if values else float("nan")


def run_one(cell, config, args):
    fixed = config["fixed"]
    N = int(fixed["n_atoms"])
    box = _cubic_box_length(N, float(fixed["density_reduced"]), float(fixed["sigma"]))
    context = {**cell, **fixed, "box_length": box, "out_dump": "dump.lammpstrj"}
    ctx = resolve_run_ctx(
        config=config, cell=cell, output_root=args.output_dir / "runs",
        template_name=config["template"], lmp_binary=args.lmp,
        dry_run=args.dry_run, skip_md=args.skip_md,
    )
    input_path = ctx.output_dir / "input.lammps"
    input_path.write_text(render_template(ctx.template_name, context))
    log_path = ctx.output_dir / "log.lammps"
    run_md(input_path, log_path, ctx)

    if ctx.dry_run:
        return

    dump = parse_lammps_dump_custom(ctx.output_dir / "dump.lammpstrj")
    velocities = dump.velocities
    if velocities is None:
        raise RuntimeError("dump has no velocity columns")
    dt_sample_fs = float(fixed["timestep"]) * float(fixed["sample_every"])
    t_max_fs = float(fixed["analysis"]["t_max_fs"])
    C = build_C(velocities, dt=dt_sample_fs, t_max=t_max_fs)
    mu = eigenvalues(C)
    S = vn_entropy(C)
    tau_int = estimate_tau_int(velocities, dt=dt_sample_fs)
    density = _parse_density_from_log(log_path)

    save_C(ctx.output_dir / "C.npz", C, metadata=cell)
    np.save(ctx.output_dir / "eigvals.npy", mu)
    write_meta(
        ctx.output_dir / "meta.json",
        {"cell": cell, "box_length_A": box, "tau_int_fs": tau_int, "density": density},
    )
    write_row(
        args.output_dir / "lj_summary.csv",
        {
            "temperature": float(cell["temperature"]),
            "seed": int(cell["seed"]),
            "density": density,
            "S": S,
            "trace_C": float(np.trace(C)),
            "tau_int_fs": tau_int,
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cells = enumerate_cells(config)
    if args.array_index is None:
        print_sweep(cells)
        return 0
    run_one(pick_cell(cells, args.array_index), config, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
