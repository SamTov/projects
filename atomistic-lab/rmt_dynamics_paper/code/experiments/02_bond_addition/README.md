# Experiment 2 — Bond Addition

LJ argon liquid at `ρ = 0.8 σ^{-3}`, `T = 80 K`, `N = 512`. Each cell
adds `k ∈ {0, 16, 64, 256, 1024}` random harmonic bonds to the
equilibrated liquid; we watch outliers appear above the MP edge and the
von Neumann entropy decrease monotonically in `k`.

Because we're using LAMMPS only (no ESPResSo), each cell runs MD twice:
**equilibrate → bond → produce**. `topology.py` picks the pair list.

## Sweep

5 k values × 3 seeds = **15 cells**.

## Reproduce

```bash
python run.py --config config.yaml                   # list sweep
python run.py --config config.yaml --array-index 0   # one cell
sbatch submit.slurm                                  # whole sweep
python analysis.py --output-dir outputs              # figures
```

## Outputs

- `outputs/runs/<tag>/equilibrated.data` — bondless equilibrated config.
- `outputs/runs/<tag>/bonded.data` — data file with `k` harmonic bonds
  (present only for `k > 0`).
- `outputs/runs/<tag>/dump.lammpstrj`, `C.npz`, `eigvals.npy`,
  `participation.npy`, `meta.json`.
- `outputs/bond_summary.csv`.
- `outputs/figures/fig_bond_spectrum.pdf`,
  `outputs/figures/fig_bond_entropy.pdf`,
  `outputs/figures/fig_bond_heatmap.pdf`.
