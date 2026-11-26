"""Which modes of C carry the ionic current, and do Na-Cl pairs move together?

Spectral decomposition of the conductivity
------------------------------------------
With z_i = +1 (Na) or -1 (Cl) and C = sum_k mu_k phi_k phi_k^T,

    z^T C z = sum_k mu_k (z . phi_k)^2,    sum_i z_i^2 C_ii = Tr C = sum_k mu_k,

so with charge weights w_k = (z . phi_k)^2 / N (which sum to one)

    1 + Delta^sigma = <mu>_z / <mu>,

the charge-weighted mean eigenvalue over the plain mean. Delta^sigma < 0
means the charge pattern lies preferentially on low-eigenvalue modes. The
null is a random +-1 vector with the same numbers of each sign, for which
the expectation of the ratio is one.

Na-Cl pair cages
----------------
The ion-centred cage matrix of the hydration study (centre plus its k
nearest partners, labelled by distance rank at each origin), here with Na
centres and Cl partners, gives how strongly and how long cation-anion pairs
co-move; the Na-Cl residence time is computed alongside.

Usage
-----
python spectral_conductivity.py list <output_base> <listfile>
python spectral_conductivity.py run <run_dir> <out_dir>
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import interval_velocities, parse_lammps_dump_custom, thermo_mean  # noqa: E402
from rmt_dynamics import build_C  # noqa: E402

_spec = importlib.util.spec_from_file_location("ion_cage", HERE.parent / "07_ion_hydration" / "run.py")
ion_cage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ion_cage)

DT_FS = 50.0
LAGS_FS = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
CAGE_LAGS_FS = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
E, KB = 1.602176634e-19, 1.380649e-23


def melted_runs(base: Path) -> list[Path]:
    rows = list(csv.DictReader(open(base / "salt_summary.csv")))
    out = []
    for r in rows:
        if float(r["density"]) < 1.6:  # melted on either branch
            out.append(base / "runs" / f"init={r['init']}_temperature={float(r['temperature']):g}_seed={r['seed']}")
    return sorted(set(out))


def analyse(run: Path, out: Path) -> None:
    m = re.search(r"init=(\w+)_temperature=([\d.]+)_seed=(\d+)", run.name)
    init, T, seed = m[1], float(m[2]), int(m[3])
    dump = parse_lammps_dump_custom(run / "dump.lammpstrj")
    types, box, P = dump.atom_types, dump.box, dump.positions
    v = interval_velocities(dump, DT_FS)
    N = v.shape[1]
    z = np.where(types == 1, 1.0, -1.0)
    mass = np.where(types == 1, 22.9898, 35.4527)
    volume = thermo_mean(run / "log.lammps", "Volume")
    rng = np.random.default_rng(seed)
    nulls = np.array([rng.permutation(z) for _ in range(50)])
    res = {"run": run.name, "init": init, "temperature": T, "seed": seed, "N": N, "volume_A3": volume,
           "series": str(run.parent.parent.name), "lags": []}
    grid = np.linspace(0, 3, 121)
    for lag in LAGS_FS:
        C = build_C(v, dt=DT_FS, t_max=lag)
        mu, Phi = np.linalg.eigh(C)
        w = (z @ Phi) ** 2 / N
        mean_mu = float(mu.mean())
        ratio = float(np.sum(mu * w) / mean_mu)
        direct = float(z @ C @ z / np.trace(C))
        wn = (nulls @ Phi) ** 2 / N
        r_null = (wn @ mu) / mean_mu
        dec = mu <= np.quantile(mu, 0.1)
        pr = 1.0 / np.sum(Phi ** 4, axis=0)
        wm = (mass @ Phi) ** 2 / float(mass @ mass)
        entry = {
            "lag_fs": lag, "ratio_spectral": ratio, "ratio_direct": direct, "Delta_sigma": ratio - 1.0,
            "null_ratio_mean": float(r_null.mean()), "null_ratio_sd": float(r_null.std()),
            "charge_weight_lowest_decile": float(w[dec].sum()),
            "null_weight_lowest_decile": float(wn[:, dec].sum(axis=1).mean()),
            "pr_lowest_decile_mean": float(pr[dec].mean()), "pr_all_mean": float(pr.mean()),
            "mass_weight_lowest_mode": float(wm[0]),
            "sigma_S_per_m": float(E * E * (z @ C @ z) * 1e25 / (volume * KB * T)),
            "sigma_NE_S_per_m": float(E * E * np.trace(C) * 1e25 / (volume * KB * T)),
        }
        if lag == 4000.0:
            x = mu / mean_mu
            o = np.argsort(x)
            entry["cum_grid"] = grid.tolist()
            entry["cum_charge"] = np.interp(grid, x[o], np.cumsum(w[o]), left=0.0, right=1.0).tolist()
            entry["cum_modes"] = np.interp(grid, x[o], np.arange(1, N + 1) / N, left=0.0, right=1.0).tolist()
            np.savez(out / f"spec_{res['series']}_{run.name}.npz", mu=mu, w=w, pr=pr)
        res["lags"].append(entry)

    # Na-centred cages of Cl partners.
    na, cl = np.flatnonzero(types == 1), np.flatnonzero(types == 2)
    stride = max(1, P.shape[0] // 200)
    r_cut, r_peak = ion_cage.first_minimum(P[::stride, na], P[::stride, cl], box)
    occ = ion_cage.occupancy(P[:, na[:64]], P[:, cl], box, r_cut)
    coord = float(occ.sum(axis=2).mean())
    tau_res = ion_cage.residence_time(occ, DT_FS)
    k = max(1, int(round(coord)))
    lag_frames = [int(round(x / DT_FS)) for x in CAGE_LAGS_FS]
    centres = na[:64]
    loc = ion_cage.local_matrices(P, centres, cl, box, k, lag_frames, 20, rng)
    nul = ion_cage.local_matrices(P, centres, cl, box, k, lag_frames, 80, rng, n_random=True)
    D_cl = {L: float(np.mean(np.sum((P[L:, cl] - P[:-L, cl]) ** 2, axis=-1)) / (6.0 * L)) for L in lag_frames}
    res["cage"] = {"r_cut": r_cut, "r_peak": r_peak, "coordination": coord, "k": k, "tau_res_fs": tau_res,
                   "lags": [{"lag_fs": lf,
                             "cage_mean": float(np.sort(np.linalg.eigvalsh(loc[L]))[:k].mean() / D_cl[L]),
                             "null_mean": float(np.sort(np.linalg.eigvalsh(nul[L]))[:k].mean() / D_cl[L])}
                            for L, lf in zip(lag_frames, CAGE_LAGS_FS)]}
    (out / f"spec_{res['series']}_{run.name}.json").write_text(json.dumps(res, indent=1))
    l4 = [x for x in res["lags"] if x["lag_fs"] == 4000.0][0]
    print(run.name, "1+D", round(l4["ratio_spectral"], 4), "direct", round(l4["ratio_direct"], 4),
          "null", round(l4["null_ratio_mean"], 4), "low-decile charge weight", round(l4["charge_weight_lowest_decile"], 3),
          "cage", [round(c["cage_mean"] / c["null_mean"], 3) for c in res["cage"]["lags"]], "tau_res", round(tau_res), flush=True)


def main() -> int:
    if sys.argv[1] == "list":
        runs = []
        for base in sys.argv[2].split(","):
            runs += melted_runs(Path(base))
        Path(sys.argv[3]).write_text("\n".join(str(r) for r in runs) + "\n")
        print(len(runs), "runs")
    else:
        out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        analyse(Path(sys.argv[2]), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
