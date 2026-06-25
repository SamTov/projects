#!/bin/bash
# Submit MD array + dependent analysis.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ARRAY_ID=$(sbatch --parsable submit.slurm)
echo "array job:    $ARRAY_ID"

ANALYSIS_ID=$(sbatch --parsable --dependency=afterok:$ARRAY_ID analyse.slurm)
echo "analyse job:  $ANALYSIS_ID  (depends on $ARRAY_ID)"
