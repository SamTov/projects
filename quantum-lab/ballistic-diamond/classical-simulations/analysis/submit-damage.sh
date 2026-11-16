#!/bin/bash
#SBATCH --job-name=bd-damage
#SBATCH --output=damage-%j.out
#SBATCH --error=damage-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Reference-slab Wigner-Seitz damage analysis of a whole sweep tree.
#
#   sbatch submit-damage.sh <sweep-root> <out-dir> [species]
#
# ~40 s and ~3 GB per impact per worker; 245 impacts take ~15 min here.
# Pure Python (numpy/scipy) -- no MPI, so activating conda is safe.

ROOT=${1:?usage: sbatch submit-damage.sh <sweep-root> <out-dir> [species]}
OUT=${2:?usage: sbatch submit-damage.sh <sweep-root> <out-dir> [species]}
SPECIES=${3:-sn}
REFS=${REFS:-/work/stovey/ballistic-diamond/references}

source ~/.bashrc 2>/dev/null || true
conda activate "${CONDA_ENV:-nanopore}"

cd "${SLURM_SUBMIT_DIR}"
python -W ignore damage_ingest.py \
    --root "${ROOT}" --references "${REFS}" --out "${OUT}" \
    --species "${SPECIES}" --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))
