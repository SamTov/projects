"""Experiment 2 — bond addition driver.

Workflow per (k_bonds, seed) cell:

1. Render and run ``lj_nvt.in.j2`` *without* a bonds file, asking LAMMPS
   to write out an equilibrated data file.
2. If k_bonds > 0, read that data file, pick k random pairs within 2σ
   (topology.py), write a bonded data file.
3. Re-render ``lj_nvt.in.j2`` pointing at the bonded data file, run the
   production NVT, dump velocities.
4. Build C, analyse.

Each cell therefore runs LAMMPS twice: equilibration then production.
"""
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

# Local topology helper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology import (  # noqa: E402
    read_lammps_data, select_random_bond_pairs, write_bonded_data,
)

from rmt_dynamics import (  # noqa: E402
    build_C, eigendecomposition, eigenvalues, participation_ratio,
    save_C, vn_entropy,
)


def _cubic_box_length(n_atoms: int, density_reduced: float, sigma: float) -> float:
    number_density_inv_vol = density_reduced / (sigma ** 3)
    return float((n_atoms / number_density_inv_vol) ** (1.0 / 3.0))


def _phase_equilibrate(ctx, config, context, seed):
    """Render + run the bondless equilibration, producing equilibrated.data."""
    eq_context = {
        **context,
        # Use the equilibration length as n_prod for this phase so we have
        # a single `run` block. No velocity dump here.
        "n_equil": 0,
        "n_prod": int(context["n_equil_actual"]),
        "write_data_file": "equilibrated.data",
        "seed": seed,
    }
    # Force "no bonds" for this phase: the template checks `bonds_file is defined`.
    eq_context.pop("bonds_file", None)
    eq_context.pop("out_dump", None)
    eq_path = ctx.output_dir / "input.equilibrate.lammps"
    eq_path.write_text(render_template(ctx.template_name, eq_context))
    run_md(eq_path, ctx.output_dir / "log.equilibrate", ctx)


def _phase_production(ctx, context, bonds_file):
    """Render + run the production with an optional bonded data file."""
    prod_context = {
        **context,
        "out_dump": "dump.lammpstrj",
        "bonds_file": bonds_file,
        "n_equil": 0,     # Already equilibrated; go straight to production.
        "n_prod": int(context["n_prod_actual"]),
    }
    prod_context.pop("write_data_file", None)
    prod_path = ctx.output_dir / "input.production.lammps"
    prod_path.write_text(render_template(ctx.template_name, prod_context))
    run_md(prod_path, ctx.output_dir / "log.production", ctx)


def run_one(cell: dict, config: dict, args: argparse.Namespace) -> None:
    fixed = config["fixed"]
    N = int(fixed["n_atoms"])
    sigma = float(fixed["sigma"])
    epsilon = float(fixed["epsilon"])
    box = _cubic_box_length(N, float(fixed["density_reduced"]), sigma)
    # Bond parameters in LAMMPS real units (kcal/mol, Å).
    k_bond_lammps = float(fixed["bond_stiffness_reduced"]) * epsilon / (sigma ** 2)
    max_distance = float(fixed["bond_max_distance_sigma"]) * sigma

    context = {
        **cell,
        **fixed,
        "box_length": box,
        # Phase-length variables are renamed so the template's single
        # `run` block can be re-used for both equilibration and production.
        "n_equil_actual": int(fixed["n_equil"]),
        "n_prod_actual": int(fixed["n_prod"]),
        # Placeholder keys the template expects in its top-level context.
        # They get overridden in each phase.
        "n_equil": 0,
        "n_prod": 0,
    }

    ctx = resolve_run_ctx(
        config=config, cell=cell,
        output_root=args.output_dir / "runs",
        template_name=config["template"],
        lmp_binary=args.lmp, dry_run=args.dry_run, skip_md=args.skip_md,
    )

    # Phase 1: equilibrate.
    _phase_equilibrate(ctx, config, context, seed=int(cell["seed"]))

    # Phase 2 prep: read equilibrated positions, generate bonded topology.
    eq_data_path = ctx.output_dir / "equilibrated.data"
    bonds_data_path = None
    bonds: list = []
    k_bonds = int(cell["k_bonds"])
    if k_bonds > 0:
        bonds_data_path = ctx.output_dir / "bonded.data"
    if not ctx.dry_run and eq_data_path.is_file():
        positions, atom_types, box_vec, mass = read_lammps_data(eq_data_path)
        bonds = select_random_bond_pairs(
            positions, box_vec, k_bonds, max_distance, seed=int(cell["seed"])
        )
        if k_bonds > 0:
            mass_by_type = {int(t): float(mass) for t in np.unique(atom_types)}
            write_bonded_data(
                bonds_data_path, positions, atom_types, box_vec, mass_by_type,
                bonds, k_bond_lammps,
            )

    # Phase 3: production.
    _phase_production(
        ctx, context,
        bonds_file=str(bonds_data_path.name) if bonds_data_path is not None else None,
    )

    if ctx.dry_run:
        return

    dump = parse_lammps_dump_custom(ctx.output_dir / "dump.lammpstrj")
    velocities = dump.velocities
    if velocities is None:
        raise RuntimeError("dump has no velocity columns")
    dt_sample_fs = float(fixed["timestep"]) * float(fixed["sample_every"])
    t_max_fs = float(fixed["analysis"]["t_max_fs"])
    C = build_C(velocities, dt=dt_sample_fs, t_max=t_max_fs)
    mu, Phi = eigendecomposition(C)
    S = vn_entropy(C)
    pr = participation_ratio(Phi)

    save_C(ctx.output_dir / "C.npz", C, metadata={**cell, "n_bonds": len(bonds)})
    np.save(ctx.output_dir / "eigvals.npy", mu)
    np.save(ctx.output_dir / "participation.npy", pr)

    # Count outlier modes above MP edge — use trace/N as sigma2, and a
    # simple heuristic for T_eff.
    sigma2 = float(np.trace(C) / N)
    from rmt_dynamics import estimate_tau_int, mp_edges
    tau_int = estimate_tau_int(velocities, dt=dt_sample_fs)
    T_eff = max(1, int(round(velocities.shape[0] * dt_sample_fs / tau_int)))
    _, lam_plus = mp_edges(N, T_eff, sigma2)
    outlier_mask = mu > lam_plus
    n_outliers = int(outlier_mask.sum())
    mean_pr_outlier = float(pr[outlier_mask].mean()) if n_outliers > 0 else float("nan")

    write_meta(
        ctx.output_dir / "meta.json",
        {
            "cell": cell, "n_bonds_actual": len(bonds),
            "box_length_A": box, "k_bond_lammps": k_bond_lammps,
            "tau_int_fs": tau_int, "T_eff": T_eff,
        },
    )
    write_row(
        args.output_dir / "bond_summary.csv",
        {
            "k_bonds": int(cell["k_bonds"]),
            "seed": int(cell["seed"]),
            "n_outliers": n_outliers,
            "mean_pr_outlier": mean_pr_outlier,
            "lambda_plus_MP": float(lam_plus),
            "S": S,
            "trace_C": float(np.trace(C)),
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
