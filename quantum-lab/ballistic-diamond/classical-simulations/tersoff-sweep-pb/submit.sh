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

module load spack/default
module load gcc/12.3.0
module load openmpi/4.1.6
module load fftw/3.3.10

source ~/.bashrc
conda activate nanopore

cd "${SLURM_SUBMIT_DIR}"

lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=1

# Use the SLURM job ID as the deterministic seed salt so re-submissions can
# be distinguished, while ${RANDOM} in the bash env of the submit host gave
# correlated seeds across fast submissions.
rseed=$(( (SLURM_JOB_ID * 2654435761) % 2147483647 ))
srun "${lmp}" -var rseed ${rseed} -in simulate.lmp
