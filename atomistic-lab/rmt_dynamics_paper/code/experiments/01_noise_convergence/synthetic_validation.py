"""Validate the Marchenko-Pastur null on synthetic independent velocities.

Independent Ornstein-Uhlenbeck (AR(1)) velocities with correlation time
tau = 5 frames are the uncorrelated reference: every off-diagonal entry of
C is sampling noise. For several lags L the script compares the spectrum
of C with Marchenko-Pastur at two sample counts: the naive
T_traj / tau_int and the lag-window count of window_T_eff.

Usage: python synthetic_validation.py <out_dir>
Writes synthetic_validation.csv and synthetic_spectrum_L50.npz.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from rmt_dynamics import (build_C, eigenvalues, estimate_tau_int, ks_distance,
                          mp_edges, window_T_eff)


def ou(n_frames, n, tau, rng):
    rho = np.exp(-1.0 / tau)
    v = np.empty((n_frames, n, 3))
    v[0] = rng.standard_normal((n, 3))
    noise = rng.standard_normal((n_frames, n, 3)) * np.sqrt(1 - rho ** 2)
    for t in range(1, n_frames):
        v[t] = rho * v[t - 1] + noise[t]
    return v


def main() -> int:
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    N, n_frames, tau = 256, 20000, 5.0
    v = ou(n_frames, N, tau, rng)
    tau_int = estimate_tau_int(v, dt=1.0)
    rows = []
    for L in (10, 25, 50, 100):
        C = build_C(v, dt=1.0, t_max=float(L))
        mu = eigenvalues(C)
        s2 = float(np.trace(C) / N)
        for label, T in (("naive", n_frames / tau_int), ("window", window_T_eff(v, 1.0, float(L)))):
            lo, hi = mp_edges(N, T, s2)
            rows.append({"lag_frames": L, "lag_over_tau": L / tau, "T_eff_kind": label, "T_eff": T, "q": N / T,
                         "KS": ks_distance(mu, N, T, s2), "mu_max_over_edge": mu.max() / hi,
                         "n_above": int((mu > hi).sum()), "n_below": int((mu < lo).sum())})
        if L == 50:
            np.savez(out / "synthetic_spectrum_L50.npz", mu=mu, sigma2=s2, N=N,
                     T_naive=n_frames / tau_int, T_window=window_T_eff(v, 1.0, 50.0))
    with open(out / "synthetic_validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
