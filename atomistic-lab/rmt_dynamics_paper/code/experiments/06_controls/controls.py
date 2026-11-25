"""Controls requested in review.

Subcommands
-----------
spectral <target> <out_dir>
    For one existing run: the spectrum of C at its production lag compared
    with (i) the lag-window Marchenko-Pastur null and (ii) a surrogate null
    built from the same trajectory with every particle's time series shifted
    circularly by an independent random offset (autocorrelations and
    non-Gaussianity kept, cross-correlations destroyed); plus a lag scan of
    the entropy deficit. Writes <out_dir>/spectral_<target>.json.

structure <state> <out_dir> <lmp>
    Re-runs one Lennard-Jones phase-sweep state with its production input,
    writing positions, and computes the Steinhardt q6 order parameter and
    g(r). Writes <out_dir>/structure_<state>.json.

Targets and states are listed in TARGETS / STATES below; `list` prints them
with their array indices.
"""
from __future__ import annotations

import json
import zlib
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import (  # noqa: E402
    interval_velocities, load_config, parse_lammps_dump_custom, render_template,
)
from rmt_dynamics import build_C, ks_distance, mp_edges, window_T_eff  # noqa: E402

BASE = Path("/work/stovey/correlation")

# name: (run directory, kind, frame spacing in fs, production lag in fs, atom selection)
TARGETS = {
    "lj_crystal_40": ("03_lj_phases/runs/init=fcc_temperature=40_seed=1", "velocity", 20.0, 2000.0, None),
    "lj_disordered_40": ("03_lj_phases/runs/init=dense_temperature=40_seed=1", "velocity", 20.0, 2000.0, None),
    "lj_liquid_60": ("03_lj_phases/runs/init=dense_temperature=60_seed=1", "velocity", 20.0, 2000.0, None),
    "lj_liquid_100": ("03_lj_phases/runs/init=dense_temperature=100_seed=1", "velocity", 20.0, 2000.0, None),
    "lj_bondfree_80": ("02_bond_addition/runs/k_bonds=0_seed=1", "velocity", 20.0, 2000.0, None),
    "water_ice_260": ("04_water_phases/runs/init=ice_temperature=260_seed=1", "position", 100.0, 4000.0, 1),
    "water_liquid_220": ("04_water_phases/runs/init=liquid_temperature=220_seed=1", "position", 100.0, 4000.0, 1),
    "water_liquid_300": ("04_water_phases/runs/init=liquid_temperature=300_seed=1", "position", 100.0, 4000.0, 1),
    "salt_crystal_1100": ("05_molten_salt/runs/init=crystal_temperature=1100_seed=1", "position", 50.0, 4000.0, None),
    "salt_liquid_900": ("05_molten_salt/runs/init=liquid_temperature=900_seed=1", "position", 50.0, 4000.0, None),
    "salt_liquid_1300": ("05_molten_salt/runs/init=liquid_temperature=1300_seed=1", "position", 50.0, 4000.0, None),
}
LAGS_FS = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
N_SURROGATES = 8

# Disordered-solid structure: production states of the LJ sweep.
STATES = (
    [("fcc", 40.0, 1), ("dense", 60.0, 1)]
    + [("dense", T, s) for T in (20.0, 30.0, 40.0, 50.0) for s in (1, 2, 3)]
    + [("dilute", 40.0, s) for s in (1, 2, 3)]
)
SIGMA = 3.405


# ------------------------------------------------------------ spectral ---
def spectral_entropy(mu: np.ndarray) -> float:
    mu = np.clip(mu, 0.0, None)
    p = mu / mu.sum()
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum() / np.log(mu.size))


def wishart_entropy(N: int, q: float, reps: int, rng) -> tuple[float, float]:
    vals = []
    for _ in range(reps):
        X = rng.standard_normal((max(N + 1, int(round(N / q))), N))
        vals.append(spectral_entropy(np.linalg.eigvalsh(X.T @ X)))
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(reps))


def circular_surrogate(v: np.ndarray, rng) -> np.ndarray:
    n_frames, n_atoms, _ = v.shape
    shift = rng.integers(0, n_frames, size=n_atoms)
    idx = (np.arange(n_frames)[:, None] + shift[None, :]) % n_frames
    return v[idx, np.arange(n_atoms)[None, :], :]


def load_velocities(run: Path, kind: str, dt: float, select_type):
    dump = parse_lammps_dump_custom(run / "dump.lammpstrj")
    v = dump.velocities if kind == "velocity" else interval_velocities(dump, dt)
    if select_type is not None:
        v = v[:, dump.atom_types == select_type, :]
    return np.ascontiguousarray(v)


def spectral(target: str, out: Path) -> None:
    rel, kind, dt, lag0, sel = TARGETS[target]
    rng = np.random.default_rng(zlib.crc32(target.encode()))
    v = load_velocities(BASE / rel, kind, dt, sel)
    n_frames, N, _ = v.shape
    res = {"target": target, "run": rel, "N": N, "n_frames": n_frames, "dt_fs": dt, "lag_fs": lag0}

    C = build_C(v, dt=dt, t_max=lag0)
    mu = np.linalg.eigvalsh(C)
    T_eff = window_T_eff(v, dt, lag0)
    q = N / T_eff
    s2 = float(np.trace(C) / N)
    lo, hi = mp_edges(N, T_eff, s2)
    S = spectral_entropy(mu)
    S_mp, S_mp_err = wishart_entropy(N, q, 16, rng)
    res.update({"T_eff": T_eff, "q": q, "S": S, "S_MP": S_mp, "S_MP_err": S_mp_err,
                "dS_MP": S_mp - S, "KS_MP": ks_distance(mu, N, T_eff, s2),
                "n_above_MP": int((mu > hi).sum()), "n_below_MP": int((mu < lo).sum()),
                "trace_over_N": s2})

    # Surrogate null from the same trajectory.
    S_sur, mu_sur = [], []
    for _ in range(N_SURROGATES):
        Cs = build_C(circular_surrogate(v, rng), dt=dt, t_max=lag0)
        m = np.linalg.eigvalsh(Cs) * (s2 / float(np.trace(Cs) / N))  # same scale
        S_sur.append(spectral_entropy(m))
        mu_sur.append(m)
    mu_sur = np.concatenate(mu_sur)
    sur_lo, sur_hi = np.quantile(mu_sur, [0.5 / N, 1 - 0.5 / N])  # one expected outlier per side per run
    res.update({"S_sur": float(np.mean(S_sur)), "S_sur_sd": float(np.std(S_sur)),
                "dS_sur": float(np.mean(S_sur) - S),
                "S_sur_minus_S_MP": float(np.mean(S_sur) - S_mp),
                "sur_edges": [float(sur_lo), float(sur_hi)], "MP_edges": [float(lo), float(hi)],
                "n_above_sur": int((mu > sur_hi).sum()), "n_below_sur": int((mu < sur_lo).sum())})
    np.save(out / f"eigs_{target}.npy", mu)
    np.save(out / f"eigs_surrogate_{target}.npy", mu_sur)

    # Lag scan.
    scan = []
    for lag in LAGS_FS:
        if lag < 2 * dt or 4.5 * n_frames * dt / lag < 1.2 * N:  # q would exceed ~0.8
            continue
        Cl = build_C(v, dt=dt, t_max=lag)
        ml = np.linalg.eigvalsh(Cl)
        Tl = window_T_eff(v, dt, lag)
        ql = N / Tl
        Sm, Se = wishart_entropy(N, ql, 8, rng)
        scan.append({"lag_fs": lag, "q": ql, "S": spectral_entropy(ml), "S_MP": Sm,
                     "dS_MP": Sm - spectral_entropy(ml), "trace_over_N": float(np.trace(Cl) / N)})
    res["lag_scan"] = scan
    (out / f"spectral_{target}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("target", "q", "dS_MP", "dS_sur", "n_above_MP", "n_above_sur",
                                          "n_below_MP", "n_below_sur")}, indent=1))


# ----------------------------------------------------------- structure ---
def q6_per_atom(pos: np.ndarray, box: np.ndarray, r_cut: float) -> np.ndarray:
    from scipy.special import sph_harm_y
    d = pos[None, :, :] - pos[:, None, :]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=-1)
    out = np.zeros(len(pos))
    for i in range(len(pos)):
        nb = np.where((r[i] > 0.1) & (r[i] < r_cut))[0]
        if nb.size == 0:
            continue
        x, y, z = d[i, nb].T
        theta = np.arccos(np.clip(z / r[i, nb], -1, 1))
        phi = np.arctan2(y, x)
        qlm = np.array([sph_harm_y(6, m, theta, phi).mean() for m in range(-6, 7)])
        out[i] = np.sqrt(4 * np.pi / 13 * np.sum(np.abs(qlm) ** 2))
    return out


def rdf(pos, box, r_max, n_bins=120):
    d = pos[None, :, :] - pos[:, None, :]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=-1)[np.triu_indices(len(pos), 1)]
    h, e = np.histogram(r, bins=n_bins, range=(0, r_max))
    c = 0.5 * (e[1:] + e[:-1])
    shell = 4 * np.pi * c ** 2 * (e[1] - e[0])
    rho = len(pos) / np.prod(box)
    return c, h / (shell * rho * len(pos) / 2)


def structure(index: int, out: Path, lmp: str) -> None:
    init, T, seed = STATES[index]
    cfg = load_config(HERE.parent / "03_lj_phases" / "config.yaml")
    fixed = cfg["fixed"]
    N = int(fixed["n_atoms"])
    rho0 = float(fixed["init_density_reduced"][init])
    work = out / f"struct_{init}_{T:g}_{seed}"
    work.mkdir(parents=True, exist_ok=True)
    ctx = {**fixed, "init": init, "temperature": T, "seed": seed,
           "box_length": (N / (rho0 / SIGMA ** 3)) ** (1 / 3),
           "out_dump": "snap.lammpstrj", "dump_columns": "x y z",
           "sample_every": 50000}
    if init == "fcc":
        ctx["lattice_a"] = SIGMA * (4.0 / rho0) ** (1 / 3)
    (work / "input.lammps").write_text(render_template("lj_nvt.in.j2", ctx))
    with open(work / "log.lammps", "w") as fh:
        subprocess.run([*shlex.split(lmp), "-log", "none", "-in", "input.lammps"], cwd=work,
                       check=True, stdout=fh, stderr=subprocess.STDOUT)
    d = parse_lammps_dump_custom(work / "snap.lammpstrj")
    frames = d.positions[1:]  # production frames
    box = d.box
    q6 = np.concatenate([q6_per_atom(np.mod(f, box), box, 1.40 * SIGMA) for f in frames])
    r, g = rdf(np.mod(frames[-1], box), box, min(box) / 2)
    res = {"init": init, "temperature": T, "seed": seed, "n_frames": len(frames),
           "q6_mean": float(q6.mean()), "q6_sd": float(q6.std()),
           "frac_crystal_like": float((q6 > 0.45).mean()),
           "q6_hist": np.histogram(q6, bins=40, range=(0, 0.7))[0].tolist(),
           "rdf_r": r.tolist(), "rdf_g": g.tolist(),
           "density_gcc": float(N * 39.948 / 6.02214076e23 / (np.prod(box) * 1e-24))}
    (out / f"structure_{init}_{T:g}_{seed}.json").write_text(json.dumps(res))
    print(init, T, seed, "q6", round(res["q6_mean"], 3), "crystal-like", round(res["frac_crystal_like"], 3),
          "rho", round(res["density_gcc"], 3))


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "list":
        for i, t in enumerate(TARGETS):
            print("spectral", i, t)
        for i, s in enumerate(STATES):
            print("structure", i, s)
        return 0
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(sys.argv[2])
    if cmd == "spectral":
        key = sys.argv[2]
        key = list(TARGETS)[int(key)] if key.isdigit() else key
        out.mkdir(parents=True, exist_ok=True)
        spectral(key, out)
    elif cmd == "structure":
        out.mkdir(parents=True, exist_ok=True)
        structure(int(sys.argv[2]), out, sys.argv[4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
