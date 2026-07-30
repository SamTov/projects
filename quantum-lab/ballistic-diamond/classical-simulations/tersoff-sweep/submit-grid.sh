#!/bin/bash

# SLURM ARRAY submission for one channel-map cell.
#
# Array task -> (grid point, repeat):
#     grid_index = TASK_ID % NGRID     (row of grid.csv, 0-based)
#     repeat     = TASK_ID / NGRID
# So tasks 0..NGRID-1 are a COMPLETE map at one repeat -- if the campaign is
# cut short, the low array indices still form a usable figure.
#
# `ensemble` is the raw task id, so every task gets its own output directory,
# and the entry point it actually used is recorded in that run's params.json.

#SBATCH --job-name=bd-sn-map
#SBATCH --output=result-%a.out
#SBATCH --error=error-%a.err
# 32 ranks, NOT a whole node: the cluster runs many small jobs, so idle
# capacity is fragmented (172 cores idle but zero nodes with 64 free).
# A 64-task request needs an entirely empty node and can wait days at
# our fairshare; 32 tasks fit existing gaps and backfill readily.
# ~131k atoms/rank, ~110 MB/rank -- comfortable for LAMMPS.
#SBATCH --ntasks=32
#SBATCH --time=10:00:00

source /etc/profile 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

module purge
module load spack/default
module load gcc/12.5.0
module load openmpi/4.1.6
module load fftw/3.3.10

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: openmpi module failed to load; current modules:" >&2
    module list 2>&1 >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=1

# NOTE: an earlier revision copied the binary to node-local /tmp to dodge
# what looked like NFS exec races.  Those 'flakes' were actually the
# RanMars seed overflow (seeds >= 9e8 abort LAMMPS), fixed separately.
# The copy also breaks multi-node allocations -- ranks on other nodes
# have no such file -- so it is deliberately gone.

NGRID=$(wc -l < grid.csv)
gi=$(( SLURM_ARRAY_TASK_ID % NGRID ))
entry=$(sed -n "$((gi + 1))p" grid.csv)
x0=${entry%,*}
y0=${entry#*,}
if [ -z "${x0}" ] || [ -z "${y0}" ]; then
    echo "ERROR: could not read grid row $((gi + 1)) from grid.csv" >&2
    exit 1
fi

# LAMMPS RanMars requires seed < 900,000,000.
rseed=$(( (SLURM_JOB_ID * 2654435761) % 899999990 + 1 ))
[ "${rseed}" -lt 1 ] && rseed=1

echo "task ${SLURM_ARRAY_TASK_ID}: grid point ${gi}/${NGRID} at (${x0}, ${y0}) lattice units"

rc=1
for attempt in 1 2 3; do
    start=${SECONDS}
    srun --export=ALL "${lmp}" \
        -var rseed ${rseed} \
        -var ensemble ${SLURM_ARRAY_TASK_ID} \
        -var x0_lat ${x0} \
        -var y0_lat ${y0} \
        -log log-${SLURM_ARRAY_TASK_ID}.lammps \
        -in simulate.lmp
    rc=$?
    [ "${rc}" -eq 0 ] && break
    if [ $((SECONDS - start)) -gt 600 ]; then
        echo "srun failed rc=${rc} after >10 min -- real failure, not retrying" >&2
        break
    fi
    echo "srun attempt ${attempt} died in $((SECONDS - start))s (rc=${rc}) -- retrying" >&2
    sleep 30
done
exit ${rc}
