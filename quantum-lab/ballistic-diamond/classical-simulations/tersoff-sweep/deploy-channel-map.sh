#!/bin/bash
# Sn channel-map campaign, 35 keV, <110>, gridded entry.
#
# CROSS design rather than a full factorial -- seven cells instead of fifteen
# for the same set of figures:
#
#   angle arm       (at 300 K) : 0, 0.5, 2, 5, 10 deg   -> channel maps vs tilt,
#                                                          depth histograms,
#                                                          damage profiles
#   temperature arm (at 0.5 deg): 0, 1100 K             -> zero-point / Debye-
#                                                          Waller calibration
#                                                          against experiment
#   (0.5 deg, 300 K is shared between the arms)
#
# Each cell: 7x7 grid over the projected channel cell x 3 repeats = 147 impacts.
# Array indices 0-48 are a complete map at repeat 0, so partial completion is
# still a usable figure.
#
# Scope overrides:
#   ANGLES / TEMPERATURES / REPEATS / GRID  e.g.
#   ANGLES="0.5" TEMPERATURES="300" REPEATS=1 ./deploy-channel-map.sh
#   PILOT=3 ./deploy-channel-map.sh          # submit only 3 tasks per cell

set -euo pipefail

ENERGY=${ENERGY:-35}
ORIENT=${ORIENT:-110}
GRID=${GRID:-7}
REPEATS=${REPEATS:-3}
PILOT=${PILOT:-0}

angle_arm_angles=(${ANGLES:-0 0.5 2 5 10})
angle_arm_temp=${ANGLE_ARM_TEMP:-300}
temp_arm_temps=(${TEMPERATURES:-0 1100})
temp_arm_angle=${TEMP_ARM_ANGLE:-0.5}

grid_csv=../grids/grid-${ORIENT}-${GRID}x${GRID}.csv
if [ ! -f "${grid_csv}" ]; then
  echo "grid ${grid_csv} missing -- generating"
  ( cd .. && python3 make_channel_grid.py --orientation "${ORIENT}" --n "${GRID}" --out grids )
fi
[ -f "${grid_csv}" ] || { echo "could not obtain grid ${grid_csv}" >&2; exit 1; }
ngrid=$(wc -l < "${grid_csv}")
ntasks=$(( ngrid * REPEATS ))

# Build the cell list: angle arm at one temperature, temperature arm at one angle.
cells=()
for a in "${angle_arm_angles[@]}"; do cells+=("${a}:${angle_arm_temp}"); done
for t in "${temp_arm_temps[@]}"; do cells+=("${temp_arm_angle}:${t}"); done

submitted=0
for cell in "${cells[@]}"; do
  angle=${cell%%:*}
  temperature=${cell#*:}
  dir=maps/orient-${ORIENT}/energy-${ENERGY}/temperature-${temperature}/angle-${angle}
  mkdir -p "${dir}"

  cp simulate.lmp submit-grid.sh "${dir}/"
  cp "${grid_csv}" "${dir}/grid.csv"

  sed -i "s/ORIENTATION/${ORIENT}/g"     "${dir}/simulate.lmp"
  sed -i "s/ENERGY_KEV/${ENERGY}/g"      "${dir}/simulate.lmp"
  sed -i "s/ANGLE_DEG/${angle}/g"        "${dir}/simulate.lmp"
  sed -i "s/TEMPERATURE/${temperature}/g" "${dir}/simulate.lmp"
  # keep map output off the randomised-entry sweep tree
  sed -i "s|/tersoff-sweep/orient-|/tersoff-sweep-maps/orient-|" "${dir}/simulate.lmp"

  if [ "${PILOT}" -gt 0 ]; then
    range="0-$(( PILOT - 1 ))"
  else
    range="0-$(( ntasks - 1 ))"
  fi
  ( cd "${dir}" && sbatch --array=${range} submit-grid.sh )
  submitted=$(( submitted + 1 ))
done

echo ""
echo "Submitted ${submitted} cells; grid=${ngrid} pts x ${REPEATS} repeats = ${ntasks} impacts/cell."
if [ "${PILOT}" -gt 0 ]; then
  echo "PILOT mode: only ${PILOT} tasks per cell."
else
  echo "Total impacts: $(( submitted * ntasks ))"
fi
echo "Monitor: squeue -u \$USER -n bd-sn-map"
