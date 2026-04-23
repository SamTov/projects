#!/bin/bash
# Deploy the formal Pb-implantation sweep:
#   energies      (keV) : 20 35 60
#   tilt angles   (deg) : 0 0.5 2
#   temperatures  (K)   : 0 300 900
#   ensembles per cond  : 100
# Total = 3 x 3 x 3 x 100 = 2700 jobs.
#
# Each (energy, angle, temperature, ensemble) gets its own working directory
# under runs/ and its own SLURM job (with an independent rseed).

set -euo pipefail

energies=(20 35 60)
angles=(0 0.5 2)
temperatures=(0 300 900)
ensembles=100

scratch_root=/work/stovey/ballistic-diamond/tersoff-sweep-pb
mkdir -p logs

for energy in "${energies[@]}"; do
  for temperature in "${temperatures[@]}"; do
    for angle in "${angles[@]}"; do
      for i in $(seq 0 $((ensembles - 1))); do
        workdir=runs/energy-${energy}/temperature-${temperature}/angle-${angle}-${i}
        mkdir -p "${workdir}"
        mkdir -p "${scratch_root}/energy-${energy}/temperature-${temperature}/angle-${angle}-${i}"

        cp simulate.lmp submit.sh "${workdir}/"

        sed -i "s/ENERGY_KEV/${energy}/g"  "${workdir}/simulate.lmp"
        sed -i "s/ANGLE_DEG/${angle}/g"    "${workdir}/simulate.lmp"
        sed -i "s/TEMPERATURE/${temperature}/g" "${workdir}/simulate.lmp"
        sed -i "s/ENSEMBLE/${i}/g"          "${workdir}/simulate.lmp"

        ( cd "${workdir}" && sbatch submit.sh )
      done
    done
  done
done
