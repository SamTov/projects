#!/bin/bash
# Submit the array of MD runs, then queue the analysis job to run after the
# whole array succeeds (afterok). The two job IDs are printed so you can
# `scancel` if needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ARRAY_ID=$(sbatch --parsable submit.slurm)
echo "array job:    $ARRAY_ID"

ANALYSIS_ID=$(sbatch --parsable --dependency=afterok:$ARRAY_ID analyse.slurm)
echo "analyse job:  $ANALYSIS_ID  (depends on $ARRAY_ID)"
