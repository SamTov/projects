# Experiment 4 — Water Phases

TIP4P/2005 water, `N_water = 512`, NPT at P = 1 atm. Temperature scan
`T ∈ {240, 260, 280, 300, 320, 350, 400, 450, 500 K}`.

`topology.py` builds the initial liquid configuration as random-orientation
waters on a cubic lattice; LAMMPS + SHAKE hold the internal geometry
rigid. The massless M site for TIP4P/2005 is synthesised at run time by
`pair_style lj/cut/tip4p/long`, so it does not appear in the data file.

## Ice Ih endpoint

The spec calls for a proton-disordered ice Ih starting configuration at
240 K. Building one correctly is outside the scope of this driver; use
an external tool (e.g. [genice](https://github.com/vitroid/GenIce),
packmol with an ice Ih template, or VMD's solvate plugin) to produce
`ice_ih.data`, then run one cell with::

    python run.py --config config.yaml --array-index 0 --ice-data ice_ih.data

The driver copies that file into the per-run directory instead of
calling `topology.build_water_data`.

## Sweep

9 temperatures × 3 seeds = **27 cells** (plus the optional ice run).

## Reproduce

```bash
python run.py --config config.yaml
python run.py --config config.yaml --array-index 0
sbatch submit.slurm
python analysis.py --output-dir outputs
```

## Outputs

- `outputs/runs/<tag>/water.data`, `dump.lammpstrj`, `log.lammps`.
- `outputs/runs/<tag>/C_oxygens.npz`, `C_all.npz`, `eigvals_oxygens.npy`,
  `eigvals_all.npy`, `meta.json`.
- `outputs/water_summary.csv`.
- `outputs/figures/fig_water_entropy.pdf`,
  `outputs/figures/fig_water_hbond_overlap.pdf`.
