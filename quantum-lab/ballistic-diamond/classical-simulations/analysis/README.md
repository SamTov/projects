# ballistic_analysis

Aggregates the per-ensemble outputs of the Sn / Pb implantation sweep into a
single HDF5 summary, and provides plotting helpers for the primary
observables (final depth, channeling fraction).

## Layout

```
analysis/
├── ballistic_analysis/
│   ├── reader.py        final.data parser (ion + data-driven surface),
│   │                    ion-only trajectory parser
│   ├── aggregate.py     walks the sweep tree, classifies each ensemble,
│   │                    writes/reads the summary HDF5
│   └── viz.py           depth histograms, channeling fraction, cell table
├── ingest.py            CLI: walk one or both sweep roots -> summary.h5
├── example_analysis.py  headless batch-figure generation
├── analyze-sweep.ipynb  interactive walkthrough of the observables
├── visualize_final.py   ZnVis 3D snapshot viewer for one final.data
└── README.md
```

## Workflow

1. Run the sweep (`tersoff-sweep[-pb]/deploy-experiment.sh` -- 81 SLURM array
   jobs per species: 3 orientations x 3 energies x 3 angles x 3 temperatures).
   Each ensemble's scratch dir ends up with: `params.json`,
   `collision-ion.lammpstraj`, `anneal-ion.lammpstraj`, `final.data`.

2. On the cluster, build the summary:

   ```bash
   cd classical-simulations/analysis
   python ingest.py \
       --sn  /work/stovey/ballistic-diamond/tersoff-sweep \
       --pb  /work/stovey/ballistic-diamond/tersoff-sweep-pb \
       --out /work/stovey/ballistic-diamond/analysis/sweep-summary.h5 \
       --workers 16
   ```

3. `scp` the summary back and explore with `analyze-sweep.ipynb`, or:

   ```python
   from ballistic_analysis import load_summary, viz
   s = load_summary("sweep-summary.h5")
   fig = viz.depth_histogram(s, "sn", by="angle_deg", E_keV=60, T_K=300)
   ```

## Summary columns (one row per ensemble)

| field        | unit  | meaning                                            |
|--------------|-------|----------------------------------------------------|
| species      | -     | "sn" or "pb"                                       |
| orientation  | -     | 100 / 110 / 111 channel axis / surface normal      |
| E_keV        | keV   | ion kinetic energy at strike                       |
| angle_deg    | deg   | polar tilt of the beam off the channel axis        |
| azimuth_deg  | deg   | per-ensemble random beam azimuth (from params.json)|
| T_K          | K     | slab temperature                                   |
| ensemble     | -     | ensemble index (SLURM array task)                  |
| x, y, z      | A     | ion final position                                 |
| surface_z    | A     | slab top surface, measured from the carbon density |
| depth        | A     | surface_z - z                                      |
| vx, vy, vz   | A/ps  | ion final velocity                                 |
| ke_eV        | eV    | ion final KE (should be thermal)                   |
| status       | -     | implanted / transmitted / reflected / lost / no_data |
| ok           | bool  | 1 iff implanted                                    |

Notes:
- **surface_z is measured per-file** (highest 1 A z-bin holding at least half
  the bulk carbon count).  Never assume the vacuum gap: it is 30 lattice
  units (~151 A), and a hardcoded 30 A once produced a 121 A depth bias.
- **transmitted** = punch-through: ion absent from final.data and its last
  trajectory frame is below the slab bottom.  A cell with many transmitted
  ions has a right-censored depth distribution (slab too thin for its tail)
  -- `viz.summarize_cells` exposes `n_transmitted` per cell.
- Statuses are decided from the ion-only collision trajectory, which is a
  few hundred KB per ensemble, so classification is cheap.

## Damage / vacancy analysis (damage_ingest.py)

Second, heavier pass over the same tree -- Wigner-Seitz defect analysis of
every `final.data` against an analytically generated reference lattice
(auto-registered; no pre-strike snapshot needed):

```bash
python damage_ingest.py --sn ... --pb ... \
    --out /work/stovey/ballistic-diamond/analysis/damage-summary.h5 --workers 16
# ~1-2 min/ensemble; --no-coordination for ~2x speed (drops sp2 count)
```

Per ensemble: `n_vac`, `n_int`, `n_lost` (sputter+transmission), `n_sp2`
(3-coordinated carbons, graphitisation proxy), plus the full **vacancy depth
array** (ragged storage: `/vac_depths` + `/vac_offsets`).  The reference
lattice is orientation-aware (100/110/111 triads read from the directory
layout).  Plot with:

```python
from ballistic_analysis.damage import load_damage
from ballistic_analysis import viz
metrics, depths = load_damage("damage-summary.h5")
viz.vacancy_depth_histogram(metrics, depths, "sn", by="angle_deg",
                            E_keV=60, T_K=300, orientation=110)
# or compare channels directly:
viz.vacancy_depth_histogram(metrics, depths, "sn", by="orientation",
                            E_keV=60, T_K=300, angle_deg=0)
```

Acceptance check before trusting production numbers: run damage_ingest on
the TEST outputs first -- a single-ion cascade should give O(50-400)
vacancies clustered along the track, near-zero elsewhere.  Thousands of
uniformly scattered "vacancies" = registration failure.  Sanity anchor:
NRT with E_d~40 eV predicts ~300-450 pairs at 60 keV; MD retains 1/3-1/2.

## Deferred (V2 candidates)

- z(t) trajectory analytics (dechanneling depth, energy-loss partition):
  the parsers are already in `reader.read_ion_trajectory`; wiring per-frame
  arrays into the HDF5 is straightforward when needed.
- Vacancy CLUSTER statistics (mono-vacancy vs pocket): one extra
  KDTree pass over the vacancy sites in `damage.analyse_damage`.
- Final-site chemistry (SnV / PbV formation): outside classical-potential
  validity -- take final.data neighbourhoods to DFT.
