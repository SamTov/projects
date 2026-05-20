# ballistic_analysis

Aggregates the per-ensemble outputs of the Sn / Pb implantation sweep into a
single HDF5 summary, and provides plotting helpers for the primary
observables (final depth, channeling fraction).

## Layout

```
analysis/
├── ballistic_analysis/
│   ├── reader.py        LAMMPS data-file parser; pulls ion final state
│   ├── aggregate.py     walks the sweep tree, builds + saves the summary
│   └── viz.py           depth histograms, channeling fraction, summary table
├── ingest.py            CLI: walk one or both sweep roots -> summary.h5
├── example_analysis.py  worked example reading summary.h5 -> PNGs
└── README.md
```

## Workflow

1. After the sweep finishes (e.g. `cd ../tersoff-sweep && ./deploy-experiment.sh`
   and same for `tersoff-sweep-pb`), every ensemble directory should contain
   a `final.data` file.

2. On the cluster, build the summary (one parallel pass):

   ```bash
   cd classical-simulations/analysis
   python ingest.py \
       --sn  /work/stovey/ballistic-diamond/tersoff-sweep \
       --pb  /work/stovey/ballistic-diamond/tersoff-sweep-pb \
       --out /work/stovey/ballistic-diamond/analysis/sweep-summary.h5 \
       --workers 16
   ```

   With 16 workers, 5400 ensembles should ingest in a few minutes.  Each row
   is the heavy ion's final (x, y, z), velocity, KE, and a derived `depth`
   (= top-of-slab - z).

3. `scp` the (~MB-scale) summary file back to your laptop and explore:

   ```bash
   python example_analysis.py sweep-summary.h5 --out figures/
   ```

   Or load it in a notebook:

   ```python
   from ballistic_analysis import load_summary, viz
   s = load_summary("sweep-summary.h5")
   fig = viz.depth_histogram(s, "sn", by="angle_deg", E_keV=60, T_K=300)
   ```

## Observables in V1

Per ensemble (one row in `/summary`):

| field        | unit  | meaning                                       |
|--------------|-------|-----------------------------------------------|
| species      | -     | "sn" or "pb"                                  |
| E_keV        | keV   | ion kinetic energy at strike                  |
| angle_deg    | deg   | tilt of slab off the [-110] channel axis      |
| T_K          | K     | equilibration / anneal temperature            |
| ensemble     | -     | ensemble index                                |
| x, y, z      | A     | ion final position                            |
| depth        | A     | top-of-slab minus z                           |
| vx, vy, vz   | A/ps  | ion final velocity                            |
| ke_eV        | eV    | ion final KE (should be ~thermal)             |
| ok           | bool  | 1 if final.data parsed cleanly                |

## Not in V1 (deferred)

- **Trajectory (z vs t) of the ion.**  Now cheap to add: as of the
  dump-strategy revision the collision and anneal dumps are ion-only
  (`dump ... ion ...`), so each ensemble produces ~120 KB of trajectory
  instead of ~20 GB.  Implementation just needs a
  `reader.read_ion_trajectory()` that parses the one-atom-per-frame dump
  and an aggregate entry that stores `(t, x, y, z)` arrays in the HDF5.
- **Defect counts** (vacancies / interstitials post-anneal).  Needs the
  pre-strike + post-anneal full-slab snapshots and a lattice-site matcher.
- **Energy-loss partition** (electronic vs nuclear).  Needs the trajectory
  and stopping-power tables together.

When you're ready for any of those, the package is structured so each is
one extra module + one or two new HDF5 datasets.
