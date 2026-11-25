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


def bond_constraint_basis(bonds, n_atoms: int) -> np.ndarray:
    """Orthonormal basis of the space spanned by bond vectors e_i - e_j.

    Its dimension is the number of independent constraints the bond graph
    imposes on relative motion, n_atoms - (number of connected components);
    redundant bonds inside a cycle do not add to it.
    """
    if len(bonds) == 0:
        return np.zeros((n_atoms, 0))
    B = np.zeros((n_atoms, len(bonds)))
    for col, bond in enumerate(bonds):
        B[int(bond[0]), col] = 1.0
        B[int(bond[1]), col] = -1.0
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, s > 1e-8 * s.max()]


def bond_subspace_overlap(Phi_low: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of the bond-constraint space captured by the modes Phi_low.

    1 means the selected eigenvectors span exactly the bonded relative
    coordinates; for random modes the expectation is Phi_low.shape[1] / N.
    """
    if basis.shape[1] == 0 or Phi_low.shape[1] == 0:
        return float("nan")
    return float(np.sum((Phi_low.T @ basis) ** 2) / basis.shape[1])


def _observable(fixed: dict) -> str:
    """'velocity' (default) or 'displacement', from the analysis block."""
    return str(fixed.get("analysis", {}).get("observable", "velocity"))


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
    eq_context.pop("data_file", None)
    eq_context.pop("out_dump", None)
    eq_path = ctx.output_dir / "input.equilibrate.lammps"
    eq_path.write_text(render_template(ctx.template_name, eq_context))
    run_md(eq_path, ctx.output_dir / "log.equilibrate", ctx)


def _phase_production(ctx, context, bonds_file):
    """Render + run the production from the equilibrated configuration.

    Every k restarts from ``equilibrated.data`` (with bonds added for k > 0),
    so the k = 0 baseline is the same liquid as the bonded runs. Velocities
    are redrawn on restart, hence the short re-thermalisation before the dump.
    """
    prod_context = {
        **context,
        "out_dump": "dump.lammpstrj",
        "dump_columns": "xu yu zu" if _observable(context) == "displacement" else "vx vy vz",
        "bonds_file": bonds_file,
        "data_file": None if bonds_file else "equilibrated.data",
        "n_equil": int(context.get("n_rethermalise", 10000)),
        "n_prod": int(context["n_prod_actual"]),
    }
    prod_context.pop("write_data_file", None)
    prod_path = ctx.output_dir / "input.production.lammps"
    prod_path.write_text(render_template(ctx.template_name, prod_context))
    run_md(prod_path, ctx.output_dir / "log.production", ctx)


def run_one(cell: dict, config: dict, args: argparse.Namespace) -> None:
    fixed = config["fixed"]
    N = int(cell.get("n_atoms", fixed["n_atoms"]))  # size scan sweeps n_atoms
    sigma = float(fixed["sigma"])
    epsilon = float(fixed["epsilon"])
    box = _cubic_box_length(N, float(fixed["density_reduced"]), sigma)
    # Bond parameters in LAMMPS real units (kcal/mol, Å).
    k_bond_lammps = float(fixed["bond_stiffness_reduced"]) * epsilon / (sigma ** 2)
    max_distance = float(fixed["bond_max_distance_sigma"]) * sigma

    context = {
        **fixed,
        **cell,
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
    dt_sample_fs = float(fixed["timestep"]) * float(fixed["sample_every"])
    if _observable(fixed) == "displacement":
        # Long lag windows need sparse sampling, and sparsely sampled
        # instantaneous velocities alias fast motion into the zero-frequency
        # estimate. The mean velocity over each sampling interval,
        # (x(t + dt) - x(t)) / dt from unwrapped positions, has the same time
        # integral without the aliasing.
        if dump.positions is None:
            raise RuntimeError("dump has no position columns")
        velocities = np.diff(dump.positions, axis=0) / dt_sample_fs
    else:
        velocities = dump.velocities
        if velocities is None:
            raise RuntimeError("dump has no velocity columns")
    t_max_fs = float(fixed["analysis"]["t_max_fs"])
    C = build_C(velocities, dt=dt_sample_fs, t_max=t_max_fs)
    mu, Phi = eigendecomposition(C)
    S = vn_entropy(C)
    pr = participation_ratio(Phi)

    save_C(ctx.output_dir / "C.npz", C, metadata={**cell, "n_bonds": len(bonds)})
    np.save(ctx.output_dir / "eigvals.npy", mu)
    np.save(ctx.output_dir / "participation.npy", pr)

    # Count modes outside the MP bulk. T_eff is the sample count of the
    # lag-window estimator (set by t_max). A bond removes one diffusive
    # degree of freedom, so it appears as a mode *below* the lower edge
    # (antisymmetric on the bonded pair); the upper edge is kept as a check.
    # sigma2 = trace/N is a first pass: it is biased low once many atoms are
    # bonded, so the cross-run analysis should prefer the k = 0 baseline.
    sigma2 = float(np.trace(C) / N)
    from rmt_dynamics import estimate_tau_int, mp_edges, window_T_eff
    tau_int = estimate_tau_int(velocities, dt=dt_sample_fs)
    T_eff = window_T_eff(velocities, dt_sample_fs, t_max_fs)
    lam_minus, lam_plus = mp_edges(N, T_eff, sigma2)
    outlier_mask = mu > lam_plus
    below_mask = mu < lam_minus
    n_outliers = int(outlier_mask.sum())
    n_below = int(below_mask.sum())
    mean_pr_outlier = float(pr[outlier_mask].mean()) if n_outliers > 0 else float("nan")

    # Compare the lowest modes with the bond graph. Atom ids in the dump are
    # sorted, so index = id - 1 matches the 0-based pairs from topology.py.
    basis = bond_constraint_basis(bonds, N)
    n_constraints = int(basis.shape[1])
    overlap_lowest = bond_subspace_overlap(Phi[:, :n_constraints], basis)
    # Pair-level view of the same signal: the relative-motion (antisymmetric)
    # mode of each bonded pair, in units of the mean diagonal of unbonded atoms.
    pairs = np.array([[int(b[0]), int(b[1])] for b in bonds], dtype=int).reshape(-1, 2)
    free = np.setdiff1d(np.arange(N), pairs.ravel())
    d_free = float(np.diag(C)[free].mean()) if free.size else float("nan")
    if len(pairs):
        anti = (C[pairs[:, 0], pairs[:, 0]] + C[pairs[:, 1], pairs[:, 1]]
                - 2.0 * C[pairs[:, 0], pairs[:, 1]]) / 2.0
        bond_anti_over_free = float(anti.mean() / d_free)
    else:
        bond_anti_over_free = float("nan")
    np.save(ctx.output_dir / "bonds.npy", np.array([[int(b[0]), int(b[1])] for b in bonds], dtype=int).reshape(-1, 2))

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
            "n_below": n_below,
            "n_bonds_actual": len(bonds),
            "n_constraints": n_constraints,
            "overlap_lowest": overlap_lowest,
            "bond_anti_over_free": bond_anti_over_free,
            "diag_free": d_free,
            "mean_pr_outlier": mean_pr_outlier,
            "lambda_plus_MP": float(lam_plus),
            "lambda_minus_MP": float(lam_minus),
            "sigma2": sigma2,
            "T_eff": T_eff,
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
