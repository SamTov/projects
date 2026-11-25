"""Nernst-Einstein deviation of molten NaCl from the block sums of C versus lag.

Usage: python salt_lag.py <temperature>   (liquid start, seed 1)
"""
import sys, json, numpy as np
sys.path.insert(0, "/home/stovey/work/projects-rmt/atomistic-lab/rmt_dynamics_paper/code/experiments")
from _common import parse_lammps_dump_custom, interval_velocities
from rmt_dynamics import build_C
T = sys.argv[1]
d = parse_lammps_dump_custom(f"/work/stovey/correlation/05_molten_salt/runs/init=liquid_temperature={T}_seed=1/dump.lammpstrj")
v = interval_velocities(d, 50.0); t = d.atom_types; z = np.where(t == 1, 1.0, -1.0)
out = []
for lag in (1000., 2000., 4000., 8000., 16000., 32000.):
    C = build_C(v, dt=50.0, t_max=lag)
    self_ = np.sum(np.diag(C)); tot = z @ C @ z
    out.append({"lag_ps": lag / 1000, "Delta_sigma": tot / self_ - 1, "trace": float(self_)})
    print(T, out[-1], flush=True)
json.dump(out, open(f"salt_lag_{T}.json", "w"))
