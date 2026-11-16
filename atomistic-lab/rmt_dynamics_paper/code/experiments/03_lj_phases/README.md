# Experiment 3 — Lennard-Jones Phases

Single-component argon LJ, NPT at P = 20 atm, `N = 500`. Temperature swept
from 20 K to 160 K, with 5 K spacing between 60 K and 140 K. The 1 atm
isobar has only a few kelvin of liquid between the triple and boiling
points; 20 atm is still subcritical and gives a liquid range of tens of
kelvin. Each temperature is run from three starting states (`init`: fcc
crystal, dense disordered, dilute gas) so that superheating and
supercooling bracket each transition instead of biasing it.

## Sweep

3 starting states × 23 temperatures × 3 seeds = **207 cells**.

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
