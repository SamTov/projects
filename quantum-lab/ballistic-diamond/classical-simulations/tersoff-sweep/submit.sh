#!/bin/bash

# SLURM ARRAY submission for one (energy, angle, temperature) cell of the
# Sn implantation sweep: 100 array tasks = 100 ensemble members.
# deploy-experiment.sh copies this into each cell directory and sbatches it.

#SBATCH --job-name=bd-sn
#SBATCH --output=result-%a.out
#SBATCH --error=error-%a.err
# No --nodes=1: whole-node requests never schedule here (the partition
# runs many small jobs, so idle capacity is fragmented across nodes).
# 32 ranks fit existing gaps; measured throughput is ~half the 64-rank
# rate, i.e. the cross-node scatter costs little beyond the rank count.
#SBATCH --ntasks=32
#SBATCH --time=14:00:00
#SBATCH --array=0-99

# Make `module` available in non-interactive batch shells, then source the
# user profile.  No conda: LAMMPS is standalone, and conda activation after
# module load shadows libmpi.so.40.
source /etc/profile 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# Load the toolchain last so its library paths win.
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

ens=${SLURM_ARRAY_TASK_ID}
# SLURM_JOB_ID is unique per array task; Knuth-hash it into a seed.
# LAMMPS RanMars (equal-style random(), fix langevin) requires
# seed < 900,000,000; clamp so rseed+7 stays below the limit.
rseed=$(( (SLURM_JOB_ID * 2654435761) % 899999990 + 1 ))
[ "${rseed}" -lt 1 ] && rseed=1

# Startup-class failures (LAMMPS dying within seconds, stochastic, node/
# fabric transients) are retried; anything that ran >10 min before dying is
# a real failure and is NOT retried.
rc=1
for attempt in 1 2 3; do
    start=${SECONDS}
    srun --export=ALL "${lmp}" \
        -var rseed ${rseed} \
        -var ensemble ${ens} \
        -log log-${ens}.lammps \
        -in simulate.lmp
    rc=$?
    [ "${rc}" -eq 0 ] && break
    if [ $((SECONDS - start)) -gt 600 ]; then
        echo "srun failed rc=${rc} after >10 min -- real failure, not retrying" >&2
        break
    fi
    echo "srun attempt ${attempt} died in $((SECONDS - start))s (rc=${rc}) -- startup flake, retrying in 30s" >&2
    sleep 30
done
exit ${rc}
