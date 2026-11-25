"""Overlap of the lowest all-atom eigenvectors with the intramolecular space.

For each run, the two O->H relative coordinates of every molecule span a
2 n_water dimensional subspace. This script measures the fraction of that
subspace captured by the 2 n_water lowest eigenvectors of the all-atom
correlation matrix (1 = exact, 2/3 = random modes), and the ratio of the
mean low and high eigenvalues.

Usage: python intramolecular_overlap.py <04_water_phases output dir>
Writes intramolecular_overlap.csv next to water_summary.csv.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    base = Path(sys.argv[1])
    rows = []
    Q = None
    for d in sorted((base / "runs").glob("init=*")):
        m = re.match(r"init=(\w+)_temperature=([\d.]+)_seed=(\d+)", d.name)
        Phi = np.load(d / "eigvecs_all.npy")
        mu = np.load(d / "eigvals_all.npy")
        nw = Phi.shape[0] // 3  # atoms ordered O, H, H per molecule
        if Q is None:
            B = np.zeros((3 * nw, 2 * nw))
            for k in range(nw):
                o = 3 * k
                B[o, 2 * k], B[o + 1, 2 * k] = 1.0, -1.0
                B[o, 2 * k + 1], B[o + 2, 2 * k + 1] = 1.0, -1.0
            Q, _ = np.linalg.qr(B)
        rows.append({
            "init": m[1], "temperature": float(m[2]), "seed": int(m[3]),
            "overlap_intramolecular": float(np.sum((Phi[:, :2 * nw].T @ Q) ** 2) / Q.shape[1]),
            "mu_low_over_high": float(mu[:2 * nw].mean() / mu[2 * nw:].mean()),
        })
    with open(base / "intramolecular_overlap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
