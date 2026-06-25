# Experiment 5 — Molten Salt Ionic Conductivity

Molten NaCl, `N = 216` ion pairs (432 atoms), Tosi–Fumi–like Born–Mayer
+ Coulomb (PPPM) via `pair_style born/coul/long`. NVT at the experimental
molten-salt density, temperature sweep `T ∈ {1100, 1200, 1400, 1600, 1800 K}`.

Two routes to `Δ^σ`:

1. **Matrix route** — block-sum decomposition of `C` gives `f_α`, `f_αβ`,
   `g_αβ`, and hence `Δ^σ` via Eq. (8)–(9) of the paper.
2. **Direct GK route** — compute the total charge current
   `J(t) = Σ_i z_i v_i(t)` and build its autocorrelation integral using
   the same `build_C` machinery (treating `J` as a single "particle" of
   three Cartesian components).

The two routes are algebraic identities, so agreement is an
implementation-correctness check. The cross-species block `C[Na, Cl]` is
separately SVD'd to look at transient ion pairs.

**Born–Mayer parameters** in `lammps_templates/molten_salt_nacl.in.j2`
are placeholders — review against Fumi & Tosi 1964 (or your preferred
reference) before production runs.

## Sweep

5 temperatures × 5 seeds = **25 cells**.

## Reproduce

```bash
python run.py --config config.yaml
python run.py --config config.yaml --array-index 0
sbatch submit.slurm
python analysis.py --output-dir outputs
```

## Outputs

- `outputs/runs/<tag>/nacl.data`, `dump.lammpstrj`, `log.lammps`.
- `outputs/runs/<tag>/C.npz`, `cross_block_svd.npy`, `meta.json`.
- `outputs/salt_summary.csv`.
- `outputs/figures/fig_salt_delta_vs_T.pdf`,
  `outputs/figures/fig_salt_crossblock_spectrum.pdf`.
