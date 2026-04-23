#!/bin/bash
# Validation sweep: fires off 6 short jobs (2 species x 3 angles) to verify that
# lattice-orient branches, electronic-stopping file I/O, and all three phases
# (warmup/collision/anneal) run end-to-end before launching the ~5400-job
# production sweep.
#
# Each job runs with aggressively shortened phases (~5 ps warmup, ~1 ps
# collision, 20 ps anneal) and a 30-minute SLURM wall time.  Output lands in
# `tests/<species>-angle-<deg>/` locally and
# `/work/stovey/ballistic-diamond/tests/<species>-angle-<deg>/` on the cluster.
#
# Coverage:
#   species      : sn, pb         (different mass / Z / stopping file)
#   angles (deg) : 0, 0.5, 2      (exercises all if/then lattice branches)
#   energy (keV) : 35             (representative mid-sweep value)
#   temperature  : 300 K          (exercises velocity-create + Langevin)
#
# If the T=0 branch matters, re-run one of the working jobs with
# `-var TEMPERATURE 0` hand-edited in its copied simulate.lmp.

set -euo pipefail

test_energy=35
test_temp=300
test_ensemble=0
angles=(0 0.5 2)

# Short run lengths (override the index defaults in simulate.lmp via -var)
warmup_steps=2500       # 5 ps
collision_steps=10000   # adaptive, bounded to <~1 ps
anneal_steps=20000      # 20 ps

scratch_root=/work/stovey/ballistic-diamond/tests
mkdir -p tests

for species_src in "sn:tersoff-sweep" "pb:tersoff-sweep-pb"; do
    species=${species_src%%:*}
    src_dir=${species_src#*:}

    for angle in "${angles[@]}"; do
        workdir=tests/${species}-angle-${angle}
        rm -rf "${workdir}"
        mkdir -p "${workdir}"
        mkdir -p "${scratch_root}/${species}-angle-${angle}"

        # --- Copy + substitute simulate.lmp ---
        cp "${src_dir}/simulate.lmp" "${workdir}/simulate.lmp"
        sed -i "s/ENERGY_KEV/${test_energy}/g"    "${workdir}/simulate.lmp"
        sed -i "s/ANGLE_DEG/${angle}/g"           "${workdir}/simulate.lmp"
        sed -i "s/TEMPERATURE/${test_temp}/g"     "${workdir}/simulate.lmp"
        sed -i "s/ENSEMBLE/${test_ensemble}/g"    "${workdir}/simulate.lmp"
        # Redirect scratch output into the tests tree so test trajectories
        # never mingle with production data.
        sed -i "s|/work/stovey/ballistic-diamond/${src_dir}|${scratch_root}/${species}-angle-${angle}|g" \
            "${workdir}/simulate.lmp"

        # --- Emit a tailored submit script (short walltime, run overrides) ---
        cat > "${workdir}/submit.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=bd-test-${species}-a${angle}
#SBATCH --output=result.out
#SBATCH --error=error.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=00:30:00

module load spack/default
module load gcc/12.3.0
module load openmpi/4.1.6
module load fftw/3.3.10

source ~/.bashrc
conda activate nanopore

cd "\${SLURM_SUBMIT_DIR}"

lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=1

rseed=\$(( (SLURM_JOB_ID * 2654435761) % 2147483647 ))
srun "\${lmp}" \\
    -var rseed \${rseed} \\
    -var warmup_steps ${warmup_steps} \\
    -var collision_steps ${collision_steps} \\
    -var anneal_steps ${anneal_steps} \\
    -in simulate.lmp
EOF
        chmod +x "${workdir}/submit.sh"

        ( cd "${workdir}" && sbatch submit.sh )
    done
done

echo ""
echo "Submitted 6 validation jobs.  Monitor with:"
echo "  squeue -u \$USER -n bd-test-sn-a0,bd-test-sn-a0.5,..."
echo "Or just: squeue -u \$USER | grep bd-test"
