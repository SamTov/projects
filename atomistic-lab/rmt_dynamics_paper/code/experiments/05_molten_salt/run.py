"""Experiment 5 — molten NaCl ionic conductivity.

For each (T, seed) cell: build a rock-salt initial condition, run NVT,
construct C over all atoms, compute Δ^σ via three routes:

  (matrix)         Δ^σ_matrix = (Σ_αβ z_α z_β g_αβ) / (Σ_α z_α² f_α)
  (direct, FFT)    via Σ_ij z_i z_j C_ij — algebraic identity with matrix
  (direct, time)   via numpy/scipy time-domain GK on J(t) (independent
                   code path; agreement with the matrix route validates
                   the FFT/Bartlett pipeline in build_C)

Plus structural cross-checks: the Na–Cl RDF gives a coordination number
and a contact graph; we correlate per-Na contact degree against the
leading singular vector of the cross-species block.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology import box_length_from_density, build_nacl_data  # noqa: E402

from rmt_dynamics import (  # noqa: E402
    build_C, contact_graph, coordination_number, estimate_tau_int,
    green_kubo_integral, save_C, time_averaged_rdf, vn_entropy,
)
from scipy.stats import spearmanr  # noqa: E402

# Species charges (e).
CHARGES = {1: +1.0, 2: -1.0}

# RDF / coordination cross-check parameters. r_first_shell is the typical
# Na-Cl first-minimum from g(r) in molten NaCl (~3.6 Å, T-dependent).
RDF_BINS = 100
RDF_FRAMES = 100         # subsample positions for RDF averaging
R_FIRST_SHELL = 3.6      # Å, NaCl first-shell cutoff


def _compute_species_blocks(C, atom_types):
    species = sorted(np.unique(atom_types).tolist())
    f_alpha = {}
    f_ab = {}
    for a in species:
        mask_a = atom_types == a
        f_alpha[a] = float(np.trace(C[mask_a, :][:, mask_a]))
        for b in species:
            mask_b = atom_types == b
            f_ab[(a, b)] = float(C[mask_a, :][:, mask_b].sum())
    g_ab = {(a, b): f_ab[(a, b)] - (f_alpha[a] if a == b else 0.0)
            for a in species for b in species}
    return species, f_alpha, f_ab, g_ab


def _delta_sigma(species, f_alpha, g_ab):
    num = 0.0
    denom = 0.0
    for a in species:
        for b in species:
            num += CHARGES[a] * CHARGES[b] * g_ab[(a, b)]
        denom += (CHARGES[a] ** 2) * f_alpha[a]
    return num / denom if denom != 0 else float("nan")


def run_one(cell, config, args):
    fixed = config["fixed"]
    n_pairs = int(fixed["n_pairs"])
    box = box_length_from_density(n_pairs, float(fixed["density"]))

    ctx = resolve_run_ctx(
        config=config, cell=cell, output_root=args.output_dir / "runs",
        template_name=config["template"], lmp_binary=args.lmp,
        dry_run=args.dry_run, skip_md=args.skip_md,
    )

    data_file = ctx.output_dir / "nacl.data"
    if not args.skip_md:
        build_nacl_data(data_file, n_pairs, box, seed=int(cell["seed"]))

    context = {
        **cell, **fixed,
        "data_file": str(data_file.name),
        "out_dump": "dump.lammpstrj",
    }
    input_path = ctx.output_dir / "input.lammps"
    input_path.write_text(render_template(ctx.template_name, context))
    run_md(input_path, ctx.output_dir / "log.lammps", ctx)

    if ctx.dry_run:
        return

    dump = parse_lammps_dump_custom(ctx.output_dir / "dump.lammpstrj")
    velocities = dump.velocities
    atom_types = dump.atom_types
    positions = dump.positions
    box = dump.box
    if velocities is None:
        raise RuntimeError("dump has no velocity columns")
    dt_sample_fs = float(fixed["timestep"]) * float(fixed["sample_every"])
    t_max_fs = float(fixed["analysis"]["t_max_fs"])

    C = build_C(velocities, dt=dt_sample_fs, t_max=t_max_fs)
    species, f_alpha, f_ab, g_ab = _compute_species_blocks(C, atom_types)

    delta_matrix = _delta_sigma(species, f_alpha, g_ab)

    # Cross-species block: singular-value decomposition.
    mask_na = atom_types == 1
    mask_cl = atom_types == 2
    cross = C[mask_na, :][:, mask_cl]
    U, s, Vt = np.linalg.svd(cross, full_matrices=False)
    np.save(ctx.output_dir / "cross_block_svd.npy", s)
    np.save(ctx.output_dir / "cross_block_U.npy", U)
    np.save(ctx.output_dir / "cross_block_Vt.npy", Vt)

    # Charge current J(t) = Σ_i z_i v_i(t).
    charges = np.array([CHARGES[int(t)] for t in atom_types], dtype=np.float64)
    J = np.einsum("tij,i->tj", velocities, charges)  # (n_frames, 3)

    # Route A: Σ_ij z_i z_j C_ij from the FFT/Bartlett build_C. Algebraic
    # identity with the matrix block sums.
    S_JJ_from_matrix = float(charges @ C @ charges)

    # Route B: textbook time-domain GK via scipy.signal.correlate +
    # trapezoidal integration. Independent code path; agreement is a real
    # validation of build_C.
    S_JJ_time_domain = green_kubo_integral(J, dt=dt_sample_fs, t_max=t_max_fs)

    # Dimensionless σ_GK / σ_NE = 1 + Δ^σ.
    sigma_ne_proportional = sum(CHARGES[a] ** 2 * f_alpha[a] for a in species)
    delta_from_C = S_JJ_from_matrix / sigma_ne_proportional - 1.0
    delta_from_time = S_JJ_time_domain / sigma_ne_proportional - 1.0

    tau_int_fs = estimate_tau_int(velocities, dt=dt_sample_fs)
    S = vn_entropy(C)

    # Structural cross-check: Na-Cl RDF, coordination number, and the
    # leading-SVD-vector / contact-degree correlation.
    rdf_centres: np.ndarray | None = None
    rdf_values: np.ndarray | None = None
    coord_number = float("nan")
    svd_contact_spearman = float("nan")
    svd_contact_pvalue = float("nan")
    if positions is not None:
        n_frames_pos = positions.shape[0]
        stride = max(1, n_frames_pos // RDF_FRAMES)
        pos_sub = positions[::stride][:RDF_FRAMES]
        na_pos = pos_sub[:, mask_na, :]
        cl_pos = pos_sub[:, mask_cl, :]
        r_max = min(float(R_FIRST_SHELL * 2), float(box.min() / 2.0 - 0.1))
        rdf_centres, rdf_values = time_averaged_rdf(
            na_pos, cl_pos, box, r_max=r_max, n_bins=RDF_BINS, same_species=False,
        )
        np.save(ctx.output_dir / "rdf_NaCl_r.npy", rdf_centres)
        np.save(ctx.output_dir / "rdf_NaCl_g.npy", rdf_values)
        volume = float(np.prod(box))
        density_cl = int(mask_cl.sum()) / volume
        coord_number = coordination_number(
            rdf_centres, rdf_values, density_cl, R_FIRST_SHELL,
        )
        # Per-frame contact graph → time-averaged contact degree per Na atom.
        degree_acc = np.zeros(int(mask_na.sum()), dtype=np.float64)
        for f in range(pos_sub.shape[0]):
            adj = contact_graph(na_pos[f], cl_pos[f], box, r_cut=R_FIRST_SHELL)
            degree_acc += adj.sum(axis=1)
        contact_degree = degree_acc / float(pos_sub.shape[0])
        np.save(ctx.output_dir / "na_contact_degree.npy", contact_degree)
        # |u_1| amplitude per Na atom from leading cross-block singular triple.
        if s.size and np.std(contact_degree) > 0:
            u1 = np.abs(U[:, 0])
            if np.std(u1) > 0:
                rho, pval = spearmanr(u1, contact_degree)
                svd_contact_spearman = float(rho)
                svd_contact_pvalue = float(pval)

    save_C(ctx.output_dir / "C.npz", C, metadata={**cell, "n_atoms": C.shape[0]})
    write_meta(
        ctx.output_dir / "meta.json",
        {
            "cell": cell, "species": species, "f_alpha": f_alpha, "f_ab": f_ab,
            "g_ab": {f"{a},{b}": v for (a, b), v in g_ab.items()},
            "tau_int_fs": tau_int_fs, "box_length_A": list(box),
            "coord_number_NaCl": coord_number,
            "svd_contact_spearman": svd_contact_spearman,
        },
    )
    write_row(
        args.output_dir / "salt_summary.csv",
        {
            "temperature": float(cell["temperature"]),
            "seed": int(cell["seed"]),
            "Delta_sigma_matrix": delta_matrix,
            "Delta_sigma_from_C_charges": delta_from_C,
            "Delta_sigma_from_time_GK": delta_from_time,
            "S_JJ_from_matrix": S_JJ_from_matrix,
            "S_JJ_time_domain": S_JJ_time_domain,
            "GK_relative_error": abs(S_JJ_from_matrix - S_JJ_time_domain) / abs(
                S_JJ_from_matrix + 1e-30
            ),
            "f_Na": f_alpha.get(1, float("nan")),
            "f_Cl": f_alpha.get(2, float("nan")),
            "tau_int_fs": tau_int_fs,
            "S": S,
            "top_cross_svd": float(s[0]) if s.size else float("nan"),
            "coord_number_NaCl": coord_number,
            "svd_contact_spearman": svd_contact_spearman,
            "svd_contact_pvalue": svd_contact_pvalue,
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
