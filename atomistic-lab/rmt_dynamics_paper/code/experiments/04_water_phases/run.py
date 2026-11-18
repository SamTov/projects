"""Experiment 4 — TIP4P/2005 water phase scan.

Per cell: build a randomised-orientation water data file at liquid density
(topology.py), run NPT at the given temperature, dump velocities, and
build two correlation matrices — one over oxygens only and one over all
atoms — as the spec requires.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    add_common_args, enumerate_cells, interval_velocities, load_config,
    parse_lammps_dump_custom, pick_cell, print_sweep, render_template,
    resolve_run_ctx, run_md, thermo_mean, write_meta, write_row,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology import build_ice_ih_data, build_water_data, liquid_box_length  # noqa: E402

from rmt_dynamics import (  # noqa: E402
    build_C, eigendecomposition, eigenvalues, estimate_tau_int,
    save_C, time_averaged_hbond_degree, vn_entropy, window_T_eff,
)
from scipy.stats import spearmanr  # noqa: E402

# Reuse the same number of frames for H-bond statistics regardless of
# trajectory length: 100 evenly-spaced frames is plenty for stable degree
# means and keeps the analysis cheap on a single node.
HBOND_FRAMES = 100


def run_one(cell, config, args):
    fixed = config["fixed"]
    n_water = int(fixed["n_water"])
    seed = int(cell["seed"])

    ctx = resolve_run_ctx(
        config=config, cell=cell,
        output_root=args.output_dir / "runs",
        template_name=config["template"],
        lmp_binary=args.lmp,
        dry_run=args.dry_run, skip_md=args.skip_md,
    )

    init = str(cell.get("init", "liquid"))
    water_data = ctx.output_dir / "water.data"
    context = {**cell, **fixed, "data_file": str(water_data.name), "out_dump": "dump.lammpstrj"}
    # The pre-melt stage belongs to the liquid start only; the template
    # runs it whenever premelt_T is set.
    context.pop("premelt_T", None)
    if args.ice_data is not None:
        import shutil
        shutil.copy(args.ice_data, water_data)
        context["barostat"] = "aniso"
    elif init == "ice":
        nx, ny, nz = (int(x) for x in fixed["ice_cells"])
        if 8 * nx * ny * nz != n_water:
            raise ValueError(f"ice_cells {fixed['ice_cells']} hold {8 * nx * ny * nz} molecules, not {n_water}")
        if not args.skip_md:
            build_ice_ih_data(water_data, nx, ny, nz, seed)
        context["barostat"] = "aniso"
    else:
        if not args.skip_md:
            box = liquid_box_length(n_water, float(fixed["initial_density"]))
            build_water_data(water_data, n_water, box, seed)
        context["barostat"] = "iso"
        context["premelt_T"] = float(fixed["premelt_T"])
        context["premelt_steps"] = int(fixed["premelt_steps"])
    input_path = ctx.output_dir / "input.lammps"
    input_path.write_text(render_template(ctx.template_name, context))
    run_md(input_path, ctx.output_dir / "log.lammps", ctx)

    if ctx.dry_run:
        return

    dump = parse_lammps_dump_custom(ctx.output_dir / "dump.lammpstrj")
    atom_types = dump.atom_types
    positions = dump.positions
    box = dump.box
    dt_sample_fs = float(fixed["timestep"]) * float(fixed["sample_every"])
    # Positions are written every 0.1 ps; interval-mean velocities keep the
    # time integral exact at that spacing.
    velocities = interval_velocities(dump, dt_sample_fs)
    t_max_fs = float(fixed["analysis"]["t_max_fs"])

    def _build_and_save(mask, name):
        sub = velocities[:, mask, :]
        Cm = build_C(sub, dt=dt_sample_fs, t_max=t_max_fs)
        mu, Phi = eigendecomposition(Cm)
        save_C(ctx.output_dir / f"C_{name}.npz", Cm, metadata={**cell, "variant": name})
        np.save(ctx.output_dir / f"eigvals_{name}.npy", mu)
        np.save(ctx.output_dir / f"eigvecs_{name}.npy", Phi)
        return Cm, mu, Phi, vn_entropy(Cm), window_T_eff(sub, dt_sample_fs, t_max_fs)

    o_mask = atom_types == 1
    h_mask = atom_types == 2
    all_mask = np.ones_like(atom_types, dtype=bool)
    C_O, mu_O, Phi_O, S_O, T_eff_O = _build_and_save(o_mask, "oxygens")
    C_all, _, _, S_all, T_eff_all = _build_and_save(all_mask, "all")
    density = thermo_mean(ctx.output_dir / "log.lammps", "Density")
    diffusion_O = float(np.trace(C_O) / C_O.shape[0])

    tau_int_fs = estimate_tau_int(velocities[:, o_mask, :], dt=dt_sample_fs)

    # H-bond cross-reference: needs positions. If the dump lacks position
    # columns, fall back to writing NaN correlations.
    hb_spearman = float("nan")
    hb_spearman_p = float("nan")
    hb_degree_mean: np.ndarray | None = None
    if positions is not None:
        # In our water topology, atoms are ordered O, H1, H2 per molecule.
        # The O atoms occupy global indices 0, 3, 6, ...; the two Hs of
        # molecule i are at indices 3i+1 and 3i+2. Translate that into
        # indices INTO the H-only sub-array.
        o_global = np.where(o_mask)[0]
        h_global = np.where(h_mask)[0]
        o_to_h_pairs = np.zeros((o_global.size, 2), dtype=np.int64)
        for i, og in enumerate(o_global):
            # The two H atoms of this molecule are the next two global indices.
            h1_global = og + 1
            h2_global = og + 2
            o_to_h_pairs[i, 0] = int(np.searchsorted(h_global, h1_global))
            o_to_h_pairs[i, 1] = int(np.searchsorted(h_global, h2_global))

        n_frames = positions.shape[0]
        stride = max(1, n_frames // HBOND_FRAMES)
        sample_frames = positions[::stride][:HBOND_FRAMES]
        o_pos = sample_frames[:, o_mask, :]
        h_pos = sample_frames[:, h_mask, :]
        hb_degree_mean = time_averaged_hbond_degree(o_pos, h_pos, o_to_h_pairs, box)
        np.save(ctx.output_dir / "hbond_degree_mean.npy", hb_degree_mean)

        # Correlate H-bond degree with top-eigenvector localisation (mean of
        # |φ_ik| over the top 5 eigenvectors of C_oxygens, per oxygen).
        top_k = min(5, Phi_O.shape[1])
        top_amp = np.mean(np.abs(Phi_O[:, -top_k:]), axis=1)
        if np.std(top_amp) > 0 and np.std(hb_degree_mean) > 0:
            rho, pval = spearmanr(top_amp, hb_degree_mean)
            hb_spearman = float(rho)
            hb_spearman_p = float(pval)

    write_meta(
        ctx.output_dir / "meta.json",
        {
            "cell": cell,
            "n_oxygens": int(o_mask.sum()),
            "tau_int_fs": tau_int_fs,
            "hbond_spearman_top5": hb_spearman,
            "hbond_spearman_p_value": hb_spearman_p,
            "hbond_mean_degree": (
                float(np.mean(hb_degree_mean)) if hb_degree_mean is not None
                else float("nan")
            ),
        },
    )
    write_row(
        args.output_dir / "water_summary.csv",
        {
            "init": init,
            "temperature": float(cell["temperature"]),
            "seed": seed,
            "density": density,
            "S_oxygens": S_O,
            "S_all": S_all,
            "S_oxygens_over_logN": S_O / float(np.log(C_O.shape[0])),
            "S_all_over_logN": S_all / float(np.log(C_all.shape[0])),
            "T_eff_oxygens": T_eff_O,
            "T_eff_all": T_eff_all,
            "N_oxygens": int(C_O.shape[0]),
            "N_all": int(C_all.shape[0]),
            "diffusion_O": diffusion_O,
            "tau_int_fs": tau_int_fs,
            "hbond_mean_degree": (
                float(np.mean(hb_degree_mean)) if hb_degree_mean is not None
                else float("nan")
            ),
            "hbond_spearman_top5": hb_spearman,
            "hbond_spearman_p_value": hb_spearman_p,
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--ice-data", type=Path, default=None,
        help="Pre-built proton-disordered ice Ih data file. If set, used "
             "instead of topology.build_water_data (for the ice endpoint).",
    )
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
