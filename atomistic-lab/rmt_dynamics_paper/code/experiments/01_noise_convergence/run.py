"""Experiment 1 — noise convergence (ideal gas).

Sweep: (n_atoms, n_prod, seed). Each array task renders the LAMMPS input
for one cell, runs the MD, builds C, and writes per-task artifacts.
Post-processing into figures + summary CSV lives in analysis.py, invoked
once the array job completes.

Usage (local dev)::

    python run.py --config config.yaml            # enumerate sweep
    python run.py --config config.yaml --array-index 0 --dry-run
    python run.py --config config.yaml --array-index 0

Usage (HPC): see submit.slurm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Keep `experiments._common` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # type: ignore[import-not-found]  # noqa: E402
    add_common_args,
    enumerate_cells,
    load_config,
    parse_lammps_dump_custom,
    pick_cell,
    print_sweep,
    render_template,
    resolve_run_ctx,
    run_md,
    write_meta,
    write_row,
)

from rmt_dynamics import (  # noqa: E402
    build_C,
    eigenvalues,
    estimate_tau_int,
    ks_distance,
    mp_edges,
    save_C,
    vn_entropy,
)


def _cubic_box_length(n_atoms: int, density_reduced: float, sigma_A: float) -> float:
    """Box edge (Å) for cubic box at reduced density ρ σ^{-3} in argon units.

    number-density (Å^-3) = density_reduced / sigma_A^3
    => box = (n_atoms / number-density)^(1/3).
    """
    number_density_inv_vol = density_reduced / (sigma_A ** 3)
    return float((n_atoms / number_density_inv_vol) ** (1.0 / 3.0))


def run_one(cell: dict, config: dict, ctx_args: argparse.Namespace) -> None:
    output_root = ctx_args.output_dir / "runs"
    fixed = config["fixed"]
    box_length = _cubic_box_length(
        n_atoms=int(cell["n_atoms"]),
        density_reduced=float(fixed["density"]),
        sigma_A=float(fixed["sigma"]),
    )
    context = {
        # Sweep variables first so they can shadow fixed if needed.
        **cell,
        **fixed,
        "box_length": box_length,
        "out_dump": "dump.lammpstrj",
    }
    ctx = resolve_run_ctx(
        config=config,
        cell=cell,
        output_root=output_root,
        template_name=config["template"],
        lmp_binary=ctx_args.lmp,
        dry_run=ctx_args.dry_run,
        skip_md=ctx_args.skip_md,
    )

    input_path = ctx.output_dir / "input.lammps"
    input_path.write_text(render_template(ctx.template_name, context))
    log_path = ctx.output_dir / "log.lammps"
    dump_path = ctx.output_dir / context["out_dump"]

    run_md(input_path, log_path, ctx)

    if ctx.dry_run:
        return

    dump = parse_lammps_dump_custom(dump_path)
    velocities = dump.velocities
    if velocities is None:
        raise RuntimeError(f"dump at {dump_path} has no velocity columns")
    dt_sample_fs = float(fixed["timestep"]) * float(fixed["sample_every"])
    t_max_fs = float(fixed["analysis"]["t_max_fs"])
    remove_mean = bool(fixed["analysis"].get("remove_mean", True))

    C = build_C(
        velocities,
        dt=dt_sample_fs,
        t_max=t_max_fs,
        component_average=True,
        remove_mean=remove_mean,
    )
    mu = eigenvalues(C)
    S = vn_entropy(C)
    tau_int = estimate_tau_int(velocities, dt=dt_sample_fs)
    n_frames = velocities.shape[0]
    T_eff = max(1, int(round(n_frames * dt_sample_fs / tau_int)))
    N = int(cell["n_atoms"])
    sigma2 = float(np.trace(C) / N)
    _, lam_plus = mp_edges(N, T_eff, sigma2)
    ks = ks_distance(mu, N, T_eff, sigma2)

    save_C(ctx.output_dir / "C.npz", C, metadata={**cell, "T_eff": T_eff})
    np.save(ctx.output_dir / "eigvals.npy", mu)
    write_meta(
        ctx.output_dir / "meta.json",
        {
            "cell": cell,
            "dt_sample_fs": dt_sample_fs,
            "t_max_fs": t_max_fs,
            "n_frames": n_frames,
            "tau_int_fs": tau_int,
            "T_eff": T_eff,
            "box_length_A": box_length,
        },
    )
    row = {
        "n_atoms": N,
        "n_prod": int(cell["n_prod"]),
        "seed": int(cell["seed"]),
        "tau_int_fs": tau_int,
        "T_eff": T_eff,
        "sigma2": sigma2,
        "lambda_max": float(mu.max()),
        "lambda_plus_MP": float(lam_plus),
        "S": S,
        "S_over_logN": S / float(np.log(N)) if N > 1 else float("nan"),
        "KS": ks,
    }
    write_row(ctx_args.output_dir / "noise_summary.csv", row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    cells = enumerate_cells(config)

    if args.array_index is None:
        print_sweep(cells)
        return 0

    cell = pick_cell(cells, args.array_index)
    run_one(cell, config, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
