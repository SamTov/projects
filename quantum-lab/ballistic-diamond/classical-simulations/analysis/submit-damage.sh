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
#   sbatch submit-damage.sh <sweep-root> <out-dir> [species] [only] [tag]
#
# ~40 s and ~3 GB per impact per worker.  Pure Python (numpy/scipy) -- no
# MPI, so activating conda is safe.
#
# On a saturated partition only short jobs backfill, so split a sweep into
# chunks with the optional path filter and give each its own summary tag:
#
#   sbatch --cpus-per-task=8 --mem=24G --time=00:15:00 submit-damage.sh \
#       <root> <out> sn temperature-300/angle-0- angle-0
#
# damage_report.py merges every summary*.csv in <out>.

ROOT=${1:?usage: sbatch submit-damage.sh <sweep-root> <out-dir> [species]}
OUT=${2:?usage: sbatch submit-damage.sh <sweep-root> <out-dir> [species]}
SPECIES=${3:-sn}
ONLY=${4:-}
TAG=${5:-}
REFS=${REFS:-/work/stovey/ballistic-diamond/references}

source ~/.bashrc 2>/dev/null || true
conda activate "${CONDA_ENV:-nanopore}"

cd "${SLURM_SUBMIT_DIR}"
extra=()
[ -n "${ONLY}" ] && extra+=(--only "${ONLY}")
[ -n "${TAG}" ] && extra+=(--summary-name "summary-${TAG}.csv")
python -W ignore damage_ingest.py \
    --root "${ROOT}" --references "${REFS}" --out "${OUT}" \
    --species "${SPECIES}" --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) "${extra[@]}"
