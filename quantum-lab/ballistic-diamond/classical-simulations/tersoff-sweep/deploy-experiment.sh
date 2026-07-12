#!/bin/bash
# Deploy the formal Sn-implantation sweep as SLURM job arrays:
#   orientations        : 100 110 111  (surface normal / channel axis)
#   energies      (keV) : 20 35 60
#   tilt angles   (deg) : 0 0.5 2      (applied to the ion velocity;
#                                       azimuth randomised per ensemble)
#   temperatures  (K)   : 0 300 900
#   ensembles per cell  : 100          (one array task each)
#
# 81 cells x 1 sbatch each = 81 array submissions, 8100 tasks total.
# Ensemble index + rseed are injected per task by submit.sh via -var;
# ORIENTATION / ENERGY_KEV / ANGLE_DEG / TEMPERATURE are sed-substituted.
#
# SCOPING: every axis can be overridden via environment variables, so small
# studies use the same machinery as the full sweep.  The Sn PoC
# (priority: does <110> alignment get Sn deeper?) is:
#
#   ORIENTATIONS="110" ENERGIES="35" TEMPERATURES="300" ENSEMBLES=15 \
#       ./deploy-experiment.sh
#
# Re-running a cell that already exists just re-submits it (dirs are reused);
# resubmitting individual failed tasks:
#   cd runs/orient-O/energy-E/temperature-T/angle-A && sbatch --array=17,42 submit.sh

set -euo pipefail

orientations=(${ORIENTATIONS:-100 110 111})
energies=(${ENERGIES:-20 35 60})
angles=(${ANGLES:-0 0.5 2})
temperatures=(${TEMPERATURES:-0 300 900})
ensembles=${ENSEMBLES:-100}

for orientation in "${orientations[@]}"; do
  for energy in "${energies[@]}"; do
    for temperature in "${temperatures[@]}"; do
      for angle in "${angles[@]}"; do
        cell=runs/orient-${orientation}/energy-${energy}/temperature-${temperature}/angle-${angle}
        mkdir -p "${cell}"

        cp simulate.lmp submit.sh "${cell}/"

        sed -i "s/ORIENTATION/${orientation}/g"     "${cell}/simulate.lmp"
        sed -i "s/ENERGY_KEV/${energy}/g"           "${cell}/simulate.lmp"
        sed -i "s/ANGLE_DEG/${angle}/g"             "${cell}/simulate.lmp"
        sed -i "s/TEMPERATURE/${temperature}/g"     "${cell}/simulate.lmp"

        ( cd "${cell}" && sbatch --array=0-$((ensembles - 1)) submit.sh )
      done
    done
  done
done

n_cells=$(( ${#orientations[@]} * ${#energies[@]} * ${#angles[@]} * ${#temperatures[@]} ))
echo ""
echo "Submitted ${n_cells} array jobs ($((n_cells * ensembles)) tasks).  Monitor with:"
echo "  squeue -u \$USER -n bd-sn"
