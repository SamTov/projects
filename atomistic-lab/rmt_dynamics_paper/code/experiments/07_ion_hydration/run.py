"""Experiment 7 (pilot) — do hydration shells move with the ion?

Per cell (salt, seed): build water + ion pairs, run TIP4P-Ew/Joung-Cheatham
MD, then analyse the correlation matrix C over oxygens and ions.

A water that stays in an ion's first shell for longer than the lag bounds
the ion-water relative coordinate, which should send one eigenvalue of C
below the noise bulk with its weight on the ion and that water. For each
lag we count sub-bulk modes, assign each to the ion whose first shell
carries most of its weight (a dynamic hydration number), and measure the
overlap of the sub-bulk subspace with the ion-shell relative coordinates.
The standard shell residence time (continuous occupancy with a 2 ps
tolerance) is computed from the same trajectory for comparison.
"""
from __future__ import annotations

import argparse
import json
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
from topology import build  # noqa: E402

from rmt_dynamics import build_C, mp_edges, window_T_eff  # noqa: E402

# Joung-Cheatham parameters for TIP4P-Ew (sigma in Angstrom, epsilon in kcal/mol).
ION_LJ = {"Li": (1.439692, 0.103988), "Na": (2.184484, 0.168437),
          "K": (2.833058, 0.279465), "Cl": (4.917761, 0.011661)}
SALTS = {"none": (None, None), "LiCl": ("Li", "Cl"), "NaCl": ("Na", "Cl"), "KCl": ("K", "Cl")}


def first_minimum(pos_ion, pos_o, box, r_max=6.0, bins=120):
    d = pos_ion[:, :, None, :] - pos_o[:, None, :, :]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=-1).ravel()
    r = r[r > 0.5]  # drop self-distances when centres are oxygens themselves
    h, e = np.histogram(r, bins=bins, range=(0.0, r_max))
    c = 0.5 * (e[1:] + e[:-1])
    g = h / (4 * np.pi * c ** 2)
    i_max = int(np.argmax(g))
    i_min = i_max + int(np.argmin(g[i_max:i_max + int(2.0 / (e[1] - e[0]))]))
    return float(c[i_min]), float(c[i_max])


def occupancy(pos_ion, pos_o, box, r_cut, chunk=2000):
    """(n_frames, n_ions, n_O) boolean first-shell occupancy."""
    out = np.empty((pos_ion.shape[0], pos_ion.shape[1], pos_o.shape[1]), dtype=bool)
    for a in range(0, pos_ion.shape[0], chunk):
        d = pos_ion[a:a + chunk, :, None, :] - pos_o[a:a + chunk, None, :, :]
        d -= box * np.round(d / box)
        out[a:a + chunk] = np.einsum("tijk,tijk->tij", d, d) < r_cut * r_cut
    return out


def residence_time(occ, dt_fs, tol_fs=2000.0):
    """Mean residence time from continuous occupancy, ignoring exits < tol."""
    tol = int(round(tol_fs / dt_fs))
    T = occ.shape[0]
    s = occ.reshape(T, -1)
    num = den = 0.0
    for col in range(s.shape[1]):
        x = s[:, col]
        if not x.any():
            continue
        # Close short gaps.
        idx = np.flatnonzero(x)
        gaps = np.diff(idx)
        y = x.copy()
        for g0, gl in zip(idx[:-1], gaps):
            if 1 < gl <= tol + 1:
                y[g0:g0 + gl] = True
        edges = np.diff(np.concatenate([[0], y.astype(np.int8), [0]]))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        ell = (stops - starts).astype(float)
        num += np.sum(ell ** 2)
        den += np.sum(ell)
    # P(stay >= t) = sum max(l - t, 0) / sum l  =>  tau = sum l^2 / (2 sum l)
    return float(num / (2 * den) * dt_fs) if den else float("nan")


def local_matrices(P, centres, cand, box, k, lags, origin_stride, rng, n_random=None):
    """Centre-plus-neighbours displacement covariance, averaged over origins.

    For every origin t0 (every `origin_stride` frames) and centre c, the k
    candidate oxygens nearest to c at t0 are labelled by distance rank and
    the (k+1) x (k+1) matrix <dr_a(tau) . dr_b(tau)> / (6 tau) is
    accumulated over the window. This is the finite-time Einstein form of C
    restricted to a cage that is defined at the start of each window, so
    exchange of shell members between windows does not wash it out.
    With `n_random`, the k partners are instead random candidates (null).
    Returns {lag: (k+1, k+1) matrix in units of A^2 per frame}.
    """
    T = P.shape[0]
    out = {}
    for L in lags:
        M = np.zeros((k + 1, k + 1))
        n = 0
        for t0 in range(0, T - L, origin_stride):
            d = P[t0, cand][None, :, :] - P[t0, centres][:, None, :]
            d -= box * np.round(d / box)
            r2 = np.einsum("cjk,cjk->cj", d, d)
            for ci, c in enumerate(centres):
                rr = r2[ci].copy()
                rr[cand == c] = np.inf  # a centre is never its own neighbour
                if n_random:
                    nb = rng.choice(np.flatnonzero(np.isfinite(rr)), size=k, replace=False)
                else:
                    nb = np.argpartition(rr, k)[:k]
                    nb = nb[np.argsort(rr[nb])]
                idx = np.concatenate([[c], cand[nb]])
                disp = P[t0 + L, idx] - P[t0, idx]
                M += disp @ disp.T
                n += 1
        out[L] = M / (n * 6.0 * L)
    return out


def analyse(ctx, cell, fixed, out_dir):
    dt = float(fixed["timestep"]) * float(fixed["sample_every"])
    an = fixed["analysis"]
    lags_fs = [float(x) for x in an["lags_fs"]]
    lags = [int(round(x / dt)) for x in lags_fs]
    dump = parse_lammps_dump_custom(ctx.output_dir / "dump.lammpstrj")
    types, box, P = dump.atom_types, dump.box, dump.positions
    o_all = np.flatnonzero(types == 1)
    rng = np.random.default_rng(int(cell["seed"]))
    stride = max(1, P.shape[0] // 200)

    # Centres: each ion species, or (pure water) a fixed sample of oxygens.
    groups = []
    if cell["salt"] == "none":
        groups.append(("water", rng.choice(o_all, size=int(an["n_water_centres"]), replace=False)))
    else:
        groups += [("cation", np.flatnonzero(types == 3)), ("anion", np.flatnonzero(types == 4))]

    D_O = {L: float(np.mean(np.sum((P[L:, o_all] - P[:-L, o_all]) ** 2, axis=-1)) / (6.0 * L)) for L in lags}
    meta = {"cell": cell, "groups": {}}
    for name, centres in groups:
        cand = o_all
        r_cut, r_peak = first_minimum(P[::stride, centres], P[::stride, cand], box)
        occ = occupancy(P[:, centres], P[:, cand], box, r_cut)
        if name == "water":  # remove self-occupancy
            for ci, c in enumerate(centres):
                occ[:, ci, np.flatnonzero(cand == c)] = False
        coord = float(occ.sum(axis=2).mean())
        tau_res = residence_time(occ, dt)
        k = max(1, int(round(coord)))
        loc = local_matrices(P, centres, cand, box, k, lags, int(an["origin_stride"]), rng)
        nul = local_matrices(P, centres, cand, box, k, lags, int(an["origin_stride"]) * 4, rng, n_random=True)
        g = {"r_cut": r_cut, "r_peak": r_peak, "coordination": coord, "k": k, "tau_res_fs": tau_res, "lags": []}
        for L, lf in zip(lags, lags_fs):
            ev = np.sort(np.linalg.eigvalsh(loc[L])) / D_O[L]
            ev0 = np.sort(np.linalg.eigvalsh(nul[L])) / D_O[L]
            row = {"salt": cell["salt"], "seed": int(cell["seed"]), "group": name, "lag_fs": lf, "k": k,
                   "coordination": coord, "r_cut": r_cut, "tau_res_fs": tau_res,
                   "cage_modes_mean": float(ev[:k].mean()), "cage_mode_min": float(ev[0]),
                   "top_mode": float(ev[-1]), "null_modes_mean": float(ev0[:k].mean()),
                   "centre_D_over_DO": float(loc[L][0, 0] / D_O[L])}
            write_row(out_dir / "cage_summary.csv", row)
            g["lags"].append({**row, "eigs": ev.tolist(), "eigs_null": ev0.tolist()})
        meta["groups"][name] = g
        print(name, "k", k, "tau_res", round(tau_res), "fs;", [(r["lag_fs"], round(r["cage_modes_mean"], 3), round(r["null_modes_mean"], 3)) for r in g["lags"]], flush=True)

    # Global matrix over oxygens and ions: sub-bulk / outlier counts.
    v = interval_velocities(dump, dt)
    sel = np.flatnonzero(types != 2)
    v_sel = np.ascontiguousarray(v[:, sel, :])
    del v
    N = len(sel)
    for lf in lags_fs:
        if 4.5 * v_sel.shape[0] * dt / lf < 1.5 * N:
            continue
        C = build_C(v_sel, dt=dt, t_max=lf)
        mu = np.linalg.eigvalsh(C)
        T_eff = window_T_eff(v_sel, dt, lf)
        s2 = float(np.mean(np.diag(C)[types[sel] == 1]))
        lo, hi = mp_edges(N, T_eff, s2)
        write_row(out_dir / "global_summary.csv", {"salt": cell["salt"], "seed": int(cell["seed"]), "lag_fs": lf,
                  "N": N, "q": N / T_eff, "n_below": int((mu < lo).sum()), "n_above": int((mu > hi).sum())})
    meta["density"] = thermo_mean(ctx.output_dir / "log.lammps", "Density")
    write_meta(ctx.output_dir / "meta.json", meta)


def run_one(cell, config, args):
    fixed = config["fixed"]
    cation, anion = SALTS[cell["salt"]]
    n_pairs = 0 if cation is None else int(fixed["n_pairs"])
    ctx = resolve_run_ctx(config=config, cell=cell, output_root=args.output_dir / "runs",
                          template_name=config["template"], lmp_binary=args.lmp,
                          dry_run=args.dry_run, skip_md=args.skip_md)
    data = ctx.output_dir / "system.data"
    if not args.skip_md:
        build(data, int(fixed["n_water"]), n_pairs, cation, anion, float(fixed["initial_density"]), int(cell["seed"]))
    cs, ce = ION_LJ[cation or "Na"]
    as_, ae = ION_LJ[anion or "Cl"]
    context = {**fixed, **cell, "data_file": data.name, "out_dump": "dump.lammpstrj",
               "cation_sigma": cs, "cation_epsilon": ce, "anion_sigma": as_, "anion_epsilon": ae}
    (ctx.output_dir / "input.lammps").write_text(render_template(config["template"], context))
    run_md(ctx.output_dir / "input.lammps", ctx.output_dir / "log.lammps", ctx)
    if ctx.dry_run:
        return
    analyse(ctx, cell, fixed, args.output_dir)


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
