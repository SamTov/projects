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
│   ├── viz.py           depth histograms, channeling fraction, cell table
│   └── damage.py        Wigner-Seitz defects against a pristine reference
│                        slab; vacancy clusters; ideal-lattice snapping
├── ingest.py            CLI: walk one or both sweep roots -> summary.h5
├── damage_ingest.py     CLI: defects per impact -> summary.csv, defect
│                        .npz files and KMC snippets
├── damage_report.py     figures + tables from damage_ingest output
├── damage_checks.py     quench-length and lattice-vibration checks
├── submit-damage.sh     SLURM wrapper for damage_ingest.py
├── fold_channel_map.py  entry point -> projected channel cell (maps)
├── plot_channel_map.py  three-panel channel-map figure from score CSVs
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

Wigner-Seitz defect analysis of every `final.data`: which lattice sites are
empty (vacancies), which carry an extra atom (interstitials), how the
vacancies cluster, and what the crystal looks like around the stopped ion.

```bash
# once per orientation: a pristine reference slab (no ion)
sed "s/ORIENTATION/110/; s/TEMPERATURE/300/" ../tersoff-sweep/make-reference.lmp > ref.lmp
# -> /work/stovey/ballistic-diamond/references/reference-orient110-T300.data

sbatch submit-damage.sh /work/stovey/ballistic-diamond/tersoff-sweep-maps \
                        /work/stovey/ballistic-diamond/analysis/damage-maps sn
python damage_report.py --results /work/stovey/ballistic-diamond/analysis/damage-maps \
                        --out figures/final
```

Output of `damage_ingest.py` (about 40 s and 3 GB per impact per worker):

| file | content |
|---|---|
| `summary.csv` | one row per impact: stop depth, entry point at the surface, vacancies / interstitials along the whole track, vacancies within 2 / 5 / 10 nm of the stopped ion, cluster counts (`V1`, `V2`, `V3p`, nearest-neighbour linkage), vacant sites next to the Sn, registration quality |
| `defects/<run>.npz` | every bulk vacancy site and interstitial atom of the impact (box frame, Å), the ion, and the displacement profile that was removed |
| `snippets/<run>.xyz` | extended-XYZ damage snippet within 10 nm of the stopped ion for lattice KMC: `Sn` at the origin, `X` = vacant site, `C` = interstitial carbon; `lattice_n` gives each site as integers in units of a/4 along the cubic axes, relative to the lattice site nearest the Sn |

The state analysed is the as-quenched MD configuration, 50 ps after the
cascade: ballistic phase and prompt recombination are over, thermally
activated annealing has not started.  That is the intended hand-over point
to KMC.

### Why a reference slab and an unwrapped registration

A production slab is ~3300 Å thick, laterally clamped by the periodic box,
pinned at the bottom and free at the top.  It is built at the 0 K lattice
constant and then heated, so it breathes along z with a period of ~70 ps and
almost no damping.  Two snapshots therefore differ by a smooth displacement
field of ~1.3 Å across the slab, which is more than half the 1.26 Å spacing
of the ⟨110⟩ planes.  A global registration (rigid shift plus linear strain)
then assigns whole planes to the wrong sites: about 10% of *undamaged* sites
came out as vacancies, against the analytic lattice and against a pristine
reference slab alike.

`damage.register_reference` removes the field by unwrapping it in 12 Å
slices from the top surface down, each slice starting from the offset of
the one above, and `damage.analyse_damage_reference` then assigns every atom
to its nearest registered site.  On the 35 keV campaign this gives an
atom–site residual of 0.061 Å (two 300 K snapshots) and 0–1 false vacancies
per ~3 million pristine sites.

Check these columns before trusting a run: `resid_median` (≈0.06 Å at
300 K), `reg_max_step` (must stay far below 0.63 Å), `control_vac` (vacancies
more than 30 nm below the ion; ~0 unless the ion stopped shallow, because
fast carbon recoils outrun a shallow ion).  Sanity anchor for totals: NRT
with E_d ≈ 40 eV gives ~300 Frenkel pairs for 35 keV of nuclear deposition;
non-channeled impacts retain 250–290 vacancies.

### Known flaw in the 2026 campaign: the slab bottom

Until commit e709bf0 the Langevin bath group contained the lateral skins at
all heights, so 19% of the pinned anchor atoms were thermostatted while
their forces were zeroed.  They random-walked out of the slab bottom
(700–950 atoms per run) and disordered the lowest ~4 nm.  Ion depths are
unaffected (ions stop within picoseconds, at least 30 nm above the bottom).
The damage analysis measures the height of this scaffold damage per run
(`scaffold_top`) and excludes it plus a 2 nm margin; `snippet_clean` flags
whether a KMC snippet stays clear of it.  New runs use the fixed inputs.

`damage.analyse_damage` (analytic reference lattice) is kept only for slabs
thin enough that the breathing displacement stays below ~0.5 Å.

## Deferred (V2 candidates)

- z(t) trajectory analytics (dechanneling depth, energy-loss partition):
  the parsers are already in `reader.read_ion_trajectory`; wiring per-frame
  arrays into the HDF5 is straightforward when needed.
- Long-time annealing of the damage snippets (vacancy migration, SnV
  formation): lattice KMC starting from `snippets/*.xyz`.
- Final-site chemistry (SnV / PbV formation): outside classical-potential
  validity -- take final.data neighbourhoods to DFT.
