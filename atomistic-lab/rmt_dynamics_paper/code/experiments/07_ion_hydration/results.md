# Experiment 7 (pilot) — hydration cages: results

Kept separate from the RMT dynamics paper (decision 2026-11-25); candidate
for its own short paper.

## Setup

TIP4P-Ew water with Joung–Cheatham ions (parameters from OpenMM
`amber14/tip4pew.xml`), 432 waters + 4 ion pairs (~0.5 mol/kg), NPT 300 K,
1 atm, 10 ns production, positions every 0.2 ps, three seeds per salt
(LiCl, NaCl, KCl) plus pure water. Cluster job 2313068; outputs in
`/work/stovey/correlation/07_ion_hydration/`.

Cage matrix: at every time origin (1 ps apart) the centre plus its k
nearest oxygens, labelled by distance rank; the (k+1)x(k+1) displacement
covariance over the lag, averaged over origins and centres, in units of the
water self-diffusion. "Cage strength" = mean of the k lowest eigenvalues
divided by the same quantity for k random (non-neighbour) waters.
Residence time: continuous first-shell occupancy with a 2 ps tolerance.

## Results (mean over 3 seeds; seed scatter <= 0.014)

| Centre | k | Residence time | Cage strength at 1, 2, 5, 10, 20, 40 ps | Minimum |
|---|---|---|---|---|
| Li+ | 4 | 104 ps | 0.39 0.34 0.30 0.28 0.29 0.32 | 10 ps |
| Na+ | 6 | 49 ps | 0.47 0.43 0.39 0.40 0.43 0.50 | 5 ps |
| K+ | 7 | 11 ps | 0.71 0.71 0.73 0.76 0.81 0.88 | 1-2 ps |
| Cl- | 7 | 11 ps | 0.77 0.76 0.76 0.77 0.81 0.87 | 2-5 ps |
| H2O | 5 | 6 ps | 0.84 0.84 0.85 0.86 0.89 0.93 | 1-2 ps |

- Ordering Li+ > Na+ > K+ ~ Cl- > water matches the independent residence
  times and the structure-maker / structure-breaker classification.
- Ion self-diffusion relative to water: Li+ 0.40-0.46, Na+ 0.37-0.40,
  K+ ~0.60, Cl- ~0.58.
- K+ and Cl- share a residence time (~11 ps) but differ in cage strength
  and timing, which the residence time alone does not resolve.
- Global matrix (oxygens + ions): at 20 ps LiCl and NaCl keep 4 extra
  sub-bulk modes (one per cation), KCl none. Not interpreted: eigenvectors
  were not saved, so ion-self-diffusion and cage contributions are not
  separated.

## Caveats and next steps

- One force field, one concentration, 12 ions per species.
- The lag of minimum cage strength is not a lifetime (it sits 5-10x below
  the residence time); a model of bounded motion plus exchange is needed to
  extract one.
- Next: save global eigenvectors; concentration and force-field dependence
  (e.g. Madrid-2019 scaled charges); divalent ions (Mg2+, Ca2+) as a strong
  test; fit cage lifetimes and compare with residence times.
- Code: `run.py` (cage analysis), `topology.py`, template
  `lammps_templates/water_ions_tip4pew.in.j2`. Synthetic check of the cage
  matrix (permanent vs exchanging cage) behaved as expected.
