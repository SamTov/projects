#!/bin/bash
# Deployment configuration for the ant cluster. Sourced by every
# submit.slurm / analyse.slurm. Values honour pre-set environment overrides.
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CODE_DIR="$(dirname "$_ENV_DIR")"

# MPI-linked LAMMPS needs the openmpi/fftw runtime. Modules only, no conda:
# activating a conda env after `module load openmpi` hides libmpi.so.40.
# The site profile and Lmod are not clean under `set -eu` (unbound
# HISTCONTROL, non-zero returns), and a failure there kills the job with an
# empty log. Relax strict mode for this block only, then check the result.
_rmt_flags="$-"
set +eu
source /etc/profile 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null
module purge
module load spack/default gcc/12.5.0 openmpi/4.1.6 fftw/3.3.10
if [[ "$_rmt_flags" == *e* ]]; then set -e; fi
if [[ "$_rmt_flags" == *u* ]]; then set -u; fi
command -v mpirun >/dev/null || { echo "openmpi module failed to load" >&2; exit 1; }

_LMP="/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export LMP_BINARY="${LMP_BINARY:-srun --export=ALL -n 1 $_LMP}"
else
    export LMP_BINARY="${LMP_BINARY:-$_LMP}"
fi
# The project venv is built once on the login node (`uv sync --extra io`).
export RMT_PYTHON="${RMT_PYTHON:-$_CODE_DIR/.venv/bin/python}"
export RMT_OUTPUT_BASE="${RMT_OUTPUT_BASE:-/work/stovey/correlation}"
