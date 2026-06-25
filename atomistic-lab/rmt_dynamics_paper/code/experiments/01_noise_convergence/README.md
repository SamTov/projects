# Experiment 1 — Convergence of the Noise Term

Ideal-gas reference: argon-parameter Langevin dynamics with pair forces
disabled. The off-diagonal structure of `C` is pure sampling noise and
should converge to a Marchenko–Pastur distribution parametrised by `N`
and `T_eff`. See `experiments.md` §1 for the hypothesis and success
criteria.

## Sweep

`n_atoms × n_prod × seed = 4 × 3 × 5 = 60` cells. Edit `config.yaml` to
change any of them; the driver enumerates the cartesian product in the
listed order.

## Reproduce

```bash
# List the sweep (prints array indices and cell contents).
python run.py --config config.yaml

# Run one cell locally with LAMMPS in PATH.
python run.py --config config.yaml --array-index 0

# HPC: submit the whole sweep.
sbatch submit.slurm

# When the array completes, generate figures + derived CSVs.
python analysis.py --output-dir outputs
```

`--dry-run` renders the LAMMPS input without executing; `--skip-md`
re-runs the analysis against an existing `dump.lammpstrj`.

## Outputs

- `outputs/runs/<cell_tag>/input.lammps`, `log.lammps`, `dump.lammpstrj`.
- `outputs/runs/<cell_tag>/C.npz` + JSON sidecar, `eigvals.npy`, `meta.json`.
- `outputs/noise_summary.csv` — one row per completed run.
- `outputs/figures/fig_noise_eigdensity.pdf`,
  `outputs/figures/fig_noise_entropy_convergence.pdf`.
