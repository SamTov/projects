#!/bin/bash
# Submit every experiment's MD array + dependent analysis job, in parallel.
# Each call to submit_chain.sh queues two jobs (array + analysis-with-
# afterok-dependency); the experiments themselves are independent so SLURM
# schedules them concurrently subject to your account / partition limits.
#
# Usage:
#   ./submit_all.sh                       # all five
#   ./submit_all.sh 01 03                 # only the listed numbers
#
# Environment variables propagate to the queued jobs (sbatch inherits the
# submitter's environment by default). Set them BEFORE calling this script:
#
#   export RMT_PYTHON="uv --project /path/to/code run python"
#   export LMP_BINARY="srun --ntasks=$SLURM_CPUS_PER_TASK lmp_mpi"
#   ./submit_all.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

declare -a all_experiments=(
    01_noise_convergence
    02_bond_addition
    03_lj_phases
    04_water_phases
    05_molten_salt
)

if [[ $# -eq 0 ]]; then
    targets=("${all_experiments[@]}")
else
    targets=()
    for arg in "$@"; do
        match=""
        for exp in "${all_experiments[@]}"; do
            if [[ "$exp" == ${arg}* || "$exp" == "$arg" ]]; then
                match="$exp"
                break
            fi
        done
        if [[ -z "$match" ]]; then
            echo "Unknown experiment selector: '$arg'. Available:" >&2
            printf '  %s\n' "${all_experiments[@]}" >&2
            exit 1
        fi
        targets+=("$match")
    done
fi

echo "Submitting: ${targets[*]}"
echo

for exp in "${targets[@]}"; do
    echo "=== $exp ==="
    (cd "$exp" && bash ./submit_chain.sh)
    echo
done

echo "All requested experiments queued."
echo "Monitor with:  squeue -u \"\$USER\""
