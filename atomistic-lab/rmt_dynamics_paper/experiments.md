# Experiments — Implementation Spec

This document is written for Claude Code (or a similar coding agent) to turn into a
runnable experiment suite. It supports the paper
*Understanding Particle Correlation: Insights from Random Matrix Theory*
(see `main.tex`, and in particular Section 4 / `input/experiments.tex`).

The paper's central object is the **atom-wise velocity correlation matrix**

    C_{ij} = ∫_0^∞ dt  <v_i(t) · v_j(0)>              (real symmetric, N × N)

where N is the number of atoms in the simulation cell. Throughout the paper
we analyse `C` through its eigenvalue spectrum, the eigenvalue density
(compared against an RMT null), and the von Neumann entropy

    S(C) = -Σ_k p_k log p_k,   p_k = μ_k / Σ_l μ_l,   {μ_k} = eig(C).

Every experiment in this document feeds the same analysis pipeline; the
variation is only in the physical system that generates the trajectory.

---

## 0. Shared infrastructure (build this first)

### 0.1 Repo layout

```
rmt_dynamics_paper/
├── main.tex                    # already exists
├── input/                      # already exists
├── figures/                    # already exists
├── experiments.md              # this file
└── code/                       # NEW
    ├── pyproject.toml
    ├── rmt_dynamics/
    │   ├── __init__.py
    │   ├── io.py               # trajectory loading
    │   ├── correlation.py      # builds C
    │   ├── spectrum.py         # eig, VN entropy, MP null
    │   ├── rmt_null.py         # Marchenko–Pastur-style reference
    │   └── plotting.py         # shared matplotlib style
    ├── experiments/
    │   ├── 01_noise_convergence/
    │   ├── 02_bond_addition/
    │   ├── 03_lj_phases/
    │   ├── 04_water_phases/
    │   └── 05_molten_salt/
    └── tests/
```

Each `experiments/NN_*` subdirectory contains:
- `run.py`          — orchestrates simulation(s) + analysis
- `config.yaml`     — all physical/numerical parameters
- `analysis.py`     — experiment-specific analysis (optional)
- `outputs/`        — trajectories, `C.npy`, eigenvalues, figures, CSVs
- `README.md`       — one-paragraph description + how to reproduce

### 0.2 Framework choices

- **Molecular dynamics**: default to LAMMPS (`lmp` via subprocess with a
  templated input file). Use ESPResSo only for experiments where we need
  programmable bond insertion mid-run (Experiment 2).
  Water can use LAMMPS with SPC/E or TIP4P/2005.
- **Python**: numpy, scipy, MDAnalysis (trajectory IO), pyyaml, tqdm, matplotlib.
  Tests with pytest.
- **Python version**: 3.11+.
- **Determinism**: every run takes a `--seed` argument, threaded through
  both the MD integrator and any numpy RNG. Record the seed in the output
  metadata.

### 0.3 The correlation module (`correlation.py`)

Implement `build_C(velocities, dt, t_max)` that:
1. Takes a velocity trajectory of shape `(n_frames, N, 3)` and timestep `dt`.
2. For each pair `(i, j)` (including `i = j`), computes
   `C_ij = Σ_τ Δτ · <v_i(t + τ) · v_j(t)>`
   where the outer average is over time origins `t`, and the sum runs over
   lag `τ ∈ [0, t_max]`.
3. Uses FFT-based correlation (`scipy.signal.fftconvolve` / Wiener–Khinchin)
   rather than the naive O(N² T²) double loop. For N up to a few thousand,
   a blocked FFT per pair is fine; above that we will revisit.
4. Returns a `(N, N)` float64 array.

Add an option to compute `C` per-component (xyz) and average, for stability.

### 0.4 The spectrum module (`spectrum.py`)

Implement:
- `eigenvalues(C)` — calls `numpy.linalg.eigvalsh` (C is symmetric).
- `vn_entropy(C)` — Eq. above, with a small-`p` cutoff (`p > 1e-12`).
- `participation_ratio(C)` — for each eigenvector `φ_k`, `PR_k = 1 / Σ_i φ_{ik}^4`.
  Used to classify modes as localised vs collective.
- `trace_normalised(C)` — returns `C / trace(C)`.

### 0.5 The RMT null (`rmt_null.py`)

Implement a Marchenko–Pastur reference for C:
- Given `N` (particles) and `T_eff` (effective number of independent time
  samples), return the MP density
  `ρ(λ) = (1 / 2πλσ²q) · √((λ_+ - λ)(λ - λ_-))`
  on `[λ_-, λ_+]` with `q = N / T_eff`.
- Add an empirical check that generates a synthetic trajectory of
  independent Ornstein–Uhlenbeck velocities and confirms the resulting
  `C` matches MP to within a goodness-of-fit tolerance. This becomes a
  test in `tests/`.
- `T_eff` is the decorrelation-adjusted sample count, roughly
  `T_traj / τ_int` where `τ_int` is the velocity autocorrelation integral
  time; the helper must estimate `τ_int` from the trajectory, not assume it.

### 0.6 Unit tests

- `test_C_symmetric`: `C = C.T` to machine precision on a random trajectory.
- `test_C_trace_equals_self_sum`: `trace(C)` matches `Σ_i f_α`
  computed from single-particle velocity autocorrelation integrals.
- `test_vn_entropy_pure_state`: identity-shaped eigenvalue distribution
  gives `log(N)`; rank-1 distribution gives `0`.
- `test_mp_null_on_white_noise`: synthetic white noise velocities
  reproduce MP density within KS distance `< 0.05`.

Run tests before running any experiment.

---

## 1. Experiment 1 — Convergence of the Noise Term

**Location**: `code/experiments/01_noise_convergence/`

### 1.1 Hypothesis

For a non-interacting particle system, the off-diagonal structure of `C`
is pure sampling noise. Its eigenvalue density should converge to a
Marchenko–Pastur-like distribution parametrised by `N` and `T_eff`, and
`S(C)` should converge (from below, toward `log(N)`) at a rate set by
`T_eff`.

### 1.2 System

Single-component Lennard-Jones gas with the pair potential **disabled**
(ideal gas). Use argon-parameter units for consistency with other
experiments.
- Box: cubic, periodic, length chosen so `ρ = 0.01 σ^{-3}` (low density).
- Particles: sweep `N ∈ {64, 256, 1024, 4096}`.
- Temperature: `T = 100 K` (fixed, well above any irrelevant scale).
- Integrator: Langevin thermostat, `Δt = 2 fs`.
- Trajectory length: sweep `T_traj ∈ {1e4, 1e5, 1e6}` steps.
- Sample velocities every 10 steps.
- Seeds: 5 independent seeds per (N, T_traj) cell.

### 1.3 Analysis

1. Build `C` for each run.
2. Compute eigenvalue density `ρ_emp(λ)` and compare against the MP null
   from `rmt_null.py` using:
   - Kolmogorov–Smirnov distance `D_KS(ρ_emp, ρ_MP)`.
   - Eigenvalue-support edge comparison: empirical `λ_max` vs MP `λ_+`.
3. Compute `S(C)` and plot it against `T_traj` on log-log axes, separately
   for each `N`. Fit the convergence exponent.
4. Compute `S(C) / log(N)` — expected to approach 1 for uncorrelated
   particles as `T_traj → ∞`.

### 1.4 Outputs

- `outputs/figures/fig_noise_eigdensity.pdf`: eigenvalue density vs MP,
  one panel per N at largest `T_traj`.
- `outputs/figures/fig_noise_entropy_convergence.pdf`: `S(C)/log(N)` vs
  `T_traj` with error bars from seeds.
- `outputs/noise_summary.csv`: columns `N, T_traj, seed, KS, lambda_max,
  lambda_plus_MP, S, S_over_logN, tau_int`.

### 1.5 Success criteria

- `D_KS < 0.05` at the largest `(N, T_traj)`.
- `S(C)/log(N) → 1` within 1 % at the largest `(N, T_traj)`.
- Scaling of `|S - log(N)|` with `T_traj` is a clean power law over at
  least one decade.

---

## 2. Experiment 2 — Bond Addition

**Location**: `code/experiments/02_bond_addition/`

### 2.1 Hypothesis

Adding harmonic bonds between randomly selected particle pairs in a
Lennard-Jones fluid injects signal into `C` that is distinguishable from
the MP noise floor: bonded pairs show up as strong positive off-diagonal
entries, the eigenvalue spectrum develops outliers above `λ_+`, and
`S(C)` decreases monotonically with bond count.

### 2.2 System

Lennard-Jones liquid in argon units, `ρ = 0.8 σ^{-3}`, `T = 80 K`, `N = 512`.
- Integrator: NVT, Nosé–Hoover, `Δt = 2 fs`.
- Trajectory length per bond count: 5e5 steps after 5e4 equilibration.
- Sample velocities every 10 steps.
- Seeds: 3.

**Bond protocol**: starting from the equilibrated liquid, add `k` harmonic
bonds between randomly-chosen pairs (no pair repeated, no self-bonds,
minimum-image distance below `2σ` at insertion time). Bond stiffness
`k_b = 100 ε/σ²`, rest length taken at current separation. Sweep
`k ∈ {0, 16, 64, 256, 1024}` (for `N = 512` that is up to ~N paired bonds).

If LAMMPS makes run-time bond insertion awkward, generate the bonded
topology first, then run a separate simulation per `k`.

### 2.3 Analysis

1. Build `C` for each `k`.
2. Count outliers: eigenvalues with `μ_k > λ_+^{MP}(N, T_eff)`.
   Compare to `k` (the number of added bonds).
3. Examine participation ratio of outlier eigenvectors — we expect
   `PR ≈ 2` (dimer-localised) for small `k`.
4. Plot `S(C)` vs `k`. Check monotonicity.
5. Plot heatmap of `|C_ij|` for one representative `(k = 0, k = 256)`.

### 2.4 Outputs

- `outputs/figures/fig_bond_spectrum.pdf`: eigenvalue histogram overlaid
  with MP edge, one panel per `k`.
- `outputs/figures/fig_bond_entropy.pdf`: `S(C)` vs `k`.
- `outputs/figures/fig_bond_heatmap.pdf`: `|C_ij|` side-by-side for
  `k = 0` and largest `k`.
- `outputs/bond_summary.csv`: `k, seed, n_outliers, S, mean_PR_outlier`.

### 2.5 Success criteria

- `n_outliers` grows approximately linearly with `k` for small `k` and
  saturates at large `k` (many-body effects).
- `S(C)` is monotonically non-increasing in `k` across seeds.
- Outlier eigenvectors for small `k` localise on 2–4 atoms.

---

## 3. Experiment 3 — Lennard-Jones Phases

**Location**: `code/experiments/03_lj_phases/`

This experiment already has a preliminary result (`figures/lj-experiment.pdf`
and `input/experiments.tex` §4.3). The task here is to redo it cleanly.

### 3.1 Hypothesis

Across the solid–liquid–gas phase diagram of a Lennard-Jones fluid, the
von Neumann entropy `S(C)` changes more sharply at first-order
transitions than the density does, and the eigenvalue spectrum shows
qualitatively different structure in each phase.

### 3.2 System

Argon-parameter LJ, `N = 512`.
- Ensemble: NPT at `P = 1 atm` for the solid–liquid–gas scan. This is
  important: the previous run appears to have been at `P = 0` (or NVT at
  liquid density), and the gas regime above `T ≈ 85 K` showed the system
  evaporating out of the cell rather than equilibrating as a gas. Either
  run NPT at `P = 1 atm` and expect a vapor phase above `T_b ≈ 87 K`, or
  replace the gas endpoint with an explicit low-density NVT gas run.
- Temperature sweep: `T ∈ {10, 20, 30, …, 150 K}` at 10 K steps, plus
  finer spacing (2 K) across `T ∈ [80, 95]` to resolve the liquid–vapor
  transition.
- Equilibration: 1e5 steps, production: 5e5 steps.
- Velocity sampling: every 10 steps.
- Seeds: 5 per temperature.

### 3.3 Analysis

1. Compute density `ρ(T)` and `S(C; T)`.
2. Compute `dS/dT` and `dρ/dT` numerically; locate peaks.
3. Report transition temperatures from both observables, with
   uncertainties propagated from seed variation.
4. Plot eigenvalue histogram at one representative T in each phase
   (e.g. 20 K solid, 60 K liquid, 100 K vapor).

### 3.4 Outputs

- `outputs/figures/fig_lj_density_entropy.pdf`: replaces the current
  `figures/lj-experiment.pdf`. (a) density with error bars,
  (b) `S(C)` with error bars, (c) `dS/dT` and `dρ/dT` overlaid on a
  third panel.
- `outputs/figures/fig_lj_spectra_by_phase.pdf`: eigenvalue histogram
  per representative temperature.
- `outputs/lj_summary.csv`: `T, seed, rho, rho_err, S, S_err`.

### 3.5 Success criteria

- Clear peak in `dS/dT` at `T_m` and `T_b` with narrower FWHM than the
  corresponding peak in `dρ/dT`.
- No spurious data points from the gas-phase evaporation artifact.
- Error bars tight enough (< 3 %) that the entropy-vs-density comparison
  is not noise-limited.

### 3.6 Update to manuscript

When this experiment finishes, replace `figures/lj-experiment.pdf` with
the new `fig_lj_density_entropy.pdf`, and update the caption and prose
in `input/experiments.tex §4.3`.

---

## 4. Experiment 4 — Water Phases

**Location**: `code/experiments/04_water_phases/`

### 4.1 Hypothesis

In a molecular liquid with hydrogen bonding, `C` carries the H-bond
network topology in its off-diagonal structure. Phase transitions
(liquid ↔ vapor in particular, and ideally ice ↔ liquid) show up in
`S(C)` and in changes to the bulk eigenvalue distribution.

### 4.2 System

TIP4P/2005 water, `N_water ∈ {512}` (i.e. 1536 atoms; start with oxygens
only for `C`, then extend).
- Ensemble: NPT, `P = 1 atm`.
- Temperatures: `T ∈ {240, 260, 280, 300, 320, 350, 400, 450, 500 K}`.
  For ice, run a separate NPT start from a proton-disordered ice Ih
  configuration at `T = 240 K`, `P = 1 atm`; verify it remains crystalline
  by RDF.
- Trajectory: 1 ns equilibration, 5 ns production, velocity sampled every
  20 fs.
- Seeds: 3.

### 4.3 Analysis

Analysis is run twice — once with `a_i = v_i` for oxygens only, and once
for all atoms.

1. Build `C` in both variants.
2. Compute `S(C)` vs T.
3. Compare `S_O(C)` (oxygens only) with `S_all(C)` — the latter mixes
   intramolecular (bond, angle) correlations with intermolecular H-bonding.
4. Extract top-k eigenvectors at `T = 300 K` and visualise which oxygens
   they localise on; cross-reference with an explicit H-bond graph
   (Luzar–Chandler criterion) computed from positions.

### 4.4 Outputs

- `outputs/figures/fig_water_entropy.pdf`: `S(C)` vs T for both variants.
- `outputs/figures/fig_water_hbond_overlap.pdf`: scatter of eigenvector
  localisation against H-bond degree.
- `outputs/water_summary.csv`.

### 4.5 Success criteria

- `S_O(C)` drops monotonically with T in the liquid range (H-bonding
  gets stronger relatively at low T), and shows a visible feature at the
  liquid–vapor boundary.
- Correlation between top-eigenvector localisation and H-bond degree is
  statistically significant (`ρ > 0.3`).

---

## 5. Experiment 5 — Molten Salt Ionic Conductivity

**Location**: `code/experiments/05_molten_salt/`

### 5.1 Hypothesis

The Nernst–Einstein deviation `Δ^σ` for a molten salt, computed via the
standard Green–Kubo route, can be read directly off the block structure
of `C` using Equations (8)–(9) of the paper. Moreover, the spectral
structure of the `N_+ × N_-` cross-species block of `C` carries
information about ion association beyond what `Δ^σ` alone reports.

### 5.2 System

Molten NaCl, `N = 216` ion pairs (432 atoms), Tosi–Fumi or Born–Mayer
parameters.
- Ensemble: NVT at experimental molten-salt density (`ρ ≈ 1.54 g/cm³`
  at `T = 1100 K`); temperature sweep `T ∈ {1100, 1200, 1400, 1600, 1800 K}`.
- Trajectory: 200 ps equilibration, 2 ns production, velocity sampled
  every 10 fs.
- Seeds: 5.

### 5.3 Analysis

1. Compute the full `C` matrix with velocities as the microscopic `a_i`.
2. From the species block sums, compute `f_α`, `f_{αβ}`, `g_{αβ}` and
   hence `Δ^σ` (with `η_α = z_α`, the ion charge).
3. Independently compute the ionic conductivity `σ` via the charge-current
   Green–Kubo integral and verify it matches
   `σ_NE · (1 + Δ^σ)`.
4. Diagonalise the `N_+ × N_-` cross-block. The leading singular vectors
   should identify transient ion pairs; compare against a cation–anion
   coordination count from the RDF.

### 5.4 Outputs

- `outputs/figures/fig_salt_delta_vs_T.pdf`: `Δ^σ(T)` from the matrix
  construction overlaid on the direct GK-charge-current calculation.
- `outputs/figures/fig_salt_crossblock_spectrum.pdf`: singular-value
  spectrum of the cross-species block at one temperature.
- `outputs/salt_summary.csv`: `T, seed, sigma_gk, sigma_NE, Delta_sigma,
  Delta_from_matrix`.

### 5.5 Success criteria

- The two routes to `σ` agree to within combined seed uncertainty at
  every T.
- `Δ^σ` is negative (typical for molten salts) and grows in magnitude
  as T drops, as expected.
- Leading cross-block singular vectors localise on ion pairs identified
  by the RDF-based coordination criterion.

---

## Execution order and dependencies

1. Build shared infrastructure (§0) and pass all tests.
2. Experiment 1 (noise convergence) — establishes the MP null that
   Experiments 2–5 rely on.
3. Experiment 2 (bond addition) — the cleanest demonstration that signal
   separates from noise.
4. Experiment 3 (LJ phases) — already partly done; needs to be redone
   cleanly before writing §4.3 for real.
5. Experiments 4 and 5 can run in parallel once 1–3 are validated.

Each experiment ends by writing a short `results.md` into its own
directory summarising what was observed, with links to the generated
figures and CSVs. Those `results.md` files are what gets folded back
into `input/experiments.tex` when we write up.

## A note on reproducibility

Every figure in the paper should be regenerable from:

```
cd code
python experiments/NN_name/run.py --seed S --config config.yaml
```

with no manual steps. The random seed used for each figure in the paper
is recorded in a `figures/provenance.json` file that `run.py` appends to.
