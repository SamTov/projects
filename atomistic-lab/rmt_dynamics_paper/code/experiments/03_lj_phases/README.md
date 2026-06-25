# Experiment 3 — Lennard-Jones Phases

Single-component argon LJ, NPT at P = 1 atm, `N = 512`. Temperature swept
from 10 K to 150 K, with finer 2 K spacing between 80 K and 94 K to
resolve the liquid–vapor transition. NPT avoids the evaporation artifact
from the preliminary run (see `experiments.md §3.2`).

## Sweep

21 temperatures × 5 seeds = **105 cells**.

## Reproduce

```bash
python run.py --config config.yaml
python run.py --config config.yaml --array-index 0
sbatch submit.slurm
python analysis.py --output-dir outputs
```

## Outputs

- `outputs/runs/<tag>/log.lammps` — density parsed from thermo output.
- `outputs/runs/<tag>/dump.lammpstrj`, `C.npz`, `eigvals.npy`, `meta.json`.
- `outputs/lj_summary.csv`.
- `outputs/figures/fig_lj_density_entropy.pdf` — replaces
  `figures/lj-experiment.pdf` once validated.
- `outputs/figures/fig_lj_spectra_by_phase.pdf`.
