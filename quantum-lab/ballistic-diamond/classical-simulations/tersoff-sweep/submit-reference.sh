#!/bin/bash
#SBATCH --job-name=bd-ref
#SBATCH --output=ref-%j.out
#SBATCH --error=ref-%j.err
#SBATCH --ntasks=32
#SBATCH --time=06:00:00

source /etc/profile 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true
module purge
module load spack/default gcc/12.5.0 openmpi/4.1.6 fftw/3.3.10
command -v mpirun >/dev/null || { echo "openmpi missing" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR}"
lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=1
rseed=$(( (SLURM_JOB_ID * 2654435761) % 899999990 + 1 ))
srun --export=ALL "${lmp}" -var rseed ${rseed} -log ref-${SLURM_JOB_ID}.lammps -in make-reference.lmp
