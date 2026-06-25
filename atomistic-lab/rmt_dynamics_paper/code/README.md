# rmt_dynamics

Shared infrastructure for the paper *Understanding Particle Correlation:
Insights from Random Matrix Theory*. Implements the atom-wise velocity
correlation matrix

```
C_{ij} = ∫_0^{t_max} dt  <v_i(t) · v_j(0)>
```

and its spectral analysis (Marchenko–Pastur null, von Neumann entropy).

## Install

Recommended: use [`uv`](https://docs.astral.sh/uv/) for a fast, isolated
environment (creates `.venv/` automatically):

```bash
cd code
uv sync --extra dev --extra io
```

Or, with plain `pip`:

```bash
cd code
pip install -e ".[dev,io]"
```

The `io` extra brings in MDAnalysis for trajectory loading; `dev` brings
in pytest. The base install is enough for everything except `load_velocities`.

## Run tests

```bash
cd code
uv run pytest          # or just: pytest, if the env is already active
```

All tests use seeded numpy RNGs; they do not require any external MD
software.

## Public API

```python
from rmt_dynamics import (
    # Correlation matrix construction (FFT/Bartlett-window)
    build_C, velocity_autocorr_integrals,
    # Spectral observables
    eigenvalues, eigendecomposition, vn_entropy,
    participation_ratio, trace_normalised,
    # Marchenko–Pastur null + autocorrelation-time estimator
    mp_density, mp_cdf, mp_edges,
    estimate_tau_int, T_eff_from_trajectory, ks_distance,
    # Trajectory & matrix IO
    load_velocities, save_C, load_C,
    # Peak / FWHM extraction (Exp 3)
    find_peak_in_window, fwhm_in_window,
    # Luzar–Chandler H-bond graph (Exp 4)
    hbond_adjacency_luzar_chandler, hbond_degree,
    time_averaged_hbond_degree,
    # Independent GK + RDF / coordination (Exp 5)
    green_kubo_integral,
    radial_distribution, time_averaged_rdf,
    coordination_number, contact_graph,
)
```

## Layout

- `rmt_dynamics/` — importable package.
- `lammps_templates/` — jinja2 input files, rendered by each experiment.
- `tests/` — pytest suite.

Per-experiment directories live in `experiments/NN_*/`; see `../experiments.md`
for the specification.

## HPC submission

Each experiment ships three files:

- `submit.slurm`   — the array job (one task per sweep cell).
- `analyse.slurm`  — the post-hoc analysis (figures + derived CSVs).
- `submit_chain.sh`— convenience: submits the array, then queues the
  analysis with `--dependency=afterok:<array_id>`.

Submit one experiment:

```bash
cd experiments/03_lj_phases
./submit_chain.sh
```

Submit the whole suite (the five experiments are independent and will run
in parallel subject to your scheduler's limits):

```bash
cd experiments
./submit_all.sh                # all five
./submit_all.sh 01 03          # only the listed numbers
```

**Three environment variables drive the SLURM scripts** — set them before
calling `sbatch` / `submit_chain.sh` / `submit_all.sh`:

```bash
# Python with rmt_dynamics installed (uv-managed env from this repo):
export RMT_PYTHON="uv --project /path/to/code run python"

# LAMMPS binary, including any MPI wrapper:
export LMP_BINARY="srun --ntasks=$SLURM_CPUS_PER_TASK lmp_mpi"

# Where bulk data lands. Defaults to /work/stovey/correlation.
export RMT_OUTPUT_BASE="/work/stovey/correlation"
```

Defaults: `python`, `lmp`, `/work/stovey/correlation`. The scripts also
leave commented-out `module load lammps/2024` / `module load python/3.11`
lines for clusters that need them.

Per-experiment outputs land under `$RMT_OUTPUT_BASE/<exp_name>/`, e.g.
`/work/stovey/correlation/03_lj_phases/runs/temperature=85_seed=1/...`.
SLURM stdout/stderr stays in `experiments/<exp_name>/slurm_logs/` next to
the scripts (a few MB; convenient for debugging).

Total sweep sizes (= SLURM array ranges):

| Experiment | Cells | Wall-time per cell (default) |
|---|---|---|
| 01_noise_convergence | 60   | 4 h |
| 02_bond_addition     | 15   | 6 h |
| 03_lj_phases         | 105  | 6 h |
| 04_water_phases      | 27   | 24 h (PPPM + SHAKE, 5 ns) |
| 05_molten_salt       | 25   | 18 h (PPPM + Coulomb, 2 ns) |
