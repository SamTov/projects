"""Experiment 3 — LJ phases (NPT scan). Computes ρ(T) and S(C, T)."""
from __future__ import annotations

import argparse
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
    build_C, eigenvalues, estimate_tau_int, save_C, vn_entropy, window_T_eff,
)


def _cubic_box_length(n_atoms: int, density_reduced: float, sigma: float) -> float:
    number_density_inv_vol = density_reduced / (sigma ** 3)
    return float((n_atoms / number_density_inv_vol) ** (1.0 / 3.0))


def _parse_density_from_log(log_path: Path) -> float:
    """Mean of the thermo 'Density' column over the production run.

    The log can hold the same thermo block twice (LAMMPS's own log and the
    captured stdout) interleaved with other output, so only rows with as many
    numeric fields as the header are kept, and the production block is the
    one reaching the largest step count after `reset_timestep 0`.
    Returns NaN if no density column is found.
    """
    blocks: list[tuple[list[str], list[list[float]]]] = []
    header: list[str] | None = None
    for line in log_path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if parts and parts[0] == "Step":
            header = parts
            blocks.append((header, []))
            continue
        if header is None or len(parts) != len(header):
            continue
        try:
            blocks[-1][1].append([float(x) for x in parts])
        except ValueError:
            continue
    blocks = [(h, rows) for h, rows in blocks if rows and "Density" in h]
    if not blocks:
        return float("nan")
    h, rows = max(blocks, key=lambda b: (b[1][-1][0] - b[1][0][0], len(b[1])))
    idx = h.index("Density")
    return float(np.mean([r[idx] for r in rows]))


def run_one(cell, config, args):
    fixed = config["fixed"]
    N = int(fixed["n_atoms"])
    init = str(cell.get("init", "dense"))
    sigma = float(fixed["sigma"])
    if "init_density_reduced" in fixed:
        density_reduced = float(fixed["init_density_reduced"][init])
    else:
        density_reduced = float(fixed["density_reduced"])
    box = _cubic_box_length(N, density_reduced, sigma)
    context = {**cell, **fixed, "box_length": box, "out_dump": "dump.lammpstrj"}
    if init == "fcc":
        n_cells = int(fixed["n_cells"])
        if 4 * n_cells ** 3 != N:
            raise ValueError(f"n_atoms = {N} does not fill {n_cells}^3 fcc cells")
        # Four atoms per cubic cell of edge a: rho = 4 / a^3.
        context["lattice_a"] = sigma * (4.0 / density_reduced) ** (1.0 / 3.0)
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
    T_eff = window_T_eff(velocities, dt_sample_fs, t_max_fs)
    density = _parse_density_from_log(log_path)
    # trace(C)/N is the mean integrated velocity autocorrelation, i.e. the
    # self-diffusion coefficient up to the factor from the dot product. It
    # tells a spectral change apart from diffusion simply switching on.
    diffusion = float(np.trace(C) / N)

    save_C(ctx.output_dir / "C.npz", C, metadata=cell)
    np.save(ctx.output_dir / "eigvals.npy", mu)
    write_meta(
        ctx.output_dir / "meta.json",
        {"cell": cell, "box_length_A": box, "tau_int_fs": tau_int, "density": density,
         "T_eff": T_eff, "t_max_fs": t_max_fs},
    )
    write_row(
        args.output_dir / "lj_summary.csv",
        {
            "init": init,
            "temperature": float(cell["temperature"]),
            "seed": int(cell["seed"]),
            "density": density,
            "S": S,
            "S_over_logN": S / float(np.log(N)),
            "diffusion": diffusion,
            "T_eff": T_eff,
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
