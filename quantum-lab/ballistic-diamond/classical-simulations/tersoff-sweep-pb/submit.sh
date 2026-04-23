#!/bin/bash

# Single-job SLURM submission for one (energy, angle, temperature, ensemble)
# combination of the Pb implantation sweep.  deploy-experiment.sh copies this
# file into each per-job working directory and sbatches it there.

#SBATCH --job-name=bd-pb
#SBATCH --output=result.out
#SBATCH --error=error.err
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=04:00:00

# Make `module` available in non-interactive batch shells, then source the
# user profile.  Conda activation is intentionally omitted: LAMMPS is a
# standalone binary, and activating conda AFTER module load clobbered
# LD_LIBRARY_PATH so that libmpi.so.40 could not be found.
source /etc/profile 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# Load the toolchain last so its library paths win.
module purge
module load spack/default
module load gcc/12.3.0
module load openmpi/4.1.6
module load fftw/3.3.10

# Fail fast if openmpi didn't actually load.
if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: openmpi module failed to load; current modules:" >&2
    module list 2>&1 >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=1

rseed=$(( (SLURM_JOB_ID * 2654435761) % 2147483647 ))
srun --export=ALL "${lmp}" -var rseed ${rseed} -in simulate.lmp
