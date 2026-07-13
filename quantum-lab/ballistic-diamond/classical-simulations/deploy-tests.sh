#!/bin/bash
# Validation sweep: 6 short jobs (2 species x 3 angles) verifying the full
# pipeline end-to-end before launching the 5400-task production sweep:
#   - velocity-tilt strike kinematics (theta + random azimuth)
#   - tersoff/zbl + zbl potentials load (SiC.tersoff.zbl must exist!)
#   - corrected electronic-stopping tables read OK
#   - fix halt / dt-reset / all three phases run through
#   - outputs land: params.json, collision-ion & anneal-ion trajectories,
#     final.data
#
# Branch coverage: Sn jobs run at 300 K (Langevin warmup path), Pb jobs at
# 0 K (minimisation warmup path).  Angle:orientation pairs are arranged so
# all three lattice branches (100/110/111) AND all three angles are hit
# across the 6 jobs.
#
# Run lengths are aggressively shortened via -var, so the collision phase
# will NOT reach the max-KE halt condition -- that's fine for a smoke test;
# the anneal-phase dt/reset keeps the still-moving ion stable.

set -euo pipefail

test_energy=35
sn_cases=("0:110" "0.5:111" "2:100")
pb_cases=("0:100" "0.5:110" "2:111")

# Short run lengths (override the index defaults in simulate.lmp via -var).
warmup_steps=2500       # 2.5 ps (or minimize at T=0, unaffected)
collision_steps=10000   # ~0.5-1 ps of cascade
anneal_steps=5000       # 5 ps

scratch_root=/work/stovey/ballistic-diamond/tests
mkdir -p tests

for species_src in "sn:tersoff-sweep:300" "pb:tersoff-sweep-pb:0"; do
    species=$(echo "${species_src}" | cut -d: -f1)
    src_dir=$(echo "${species_src}" | cut -d: -f2)
    test_temp=$(echo "${species_src}" | cut -d: -f3)

    cases_var=${species}_cases[@]
    for case in "${!cases_var}"; do
        angle=${case%%:*}
        orientation=${case#*:}
        tag=${species}-o${orientation}-a${angle}
        workdir=tests/${tag}
        rm -rf "${workdir}"
        mkdir -p "${workdir}"
        # Wipe scratch so stale outputs from earlier revisions never mix in.
        rm -rf "${scratch_root}/${tag}"
        mkdir -p "${scratch_root}/${tag}"

        # --- Copy + substitute simulate.lmp ---
        cp "${src_dir}/simulate.lmp" "${workdir}/simulate.lmp"
        sed -i "s/ORIENTATION/${orientation}/g"   "${workdir}/simulate.lmp"
        sed -i "s/ENERGY_KEV/${test_energy}/g"    "${workdir}/simulate.lmp"
        sed -i "s/ANGLE_DEG/${angle}/g"           "${workdir}/simulate.lmp"
        sed -i "s/TEMPERATURE/${test_temp}/g"     "${workdir}/simulate.lmp"
        # Redirect scratch output into the tests tree so test trajectories
        # never mingle with production data.
        sed -i "s|/work/stovey/ballistic-diamond/${src_dir}|${scratch_root}/${tag}|g" \
            "${workdir}/simulate.lmp"

        # --- Emit a tailored submit script (short walltime, run overrides) ---
        cat > "${workdir}/submit.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=bd-test-${tag}
#SBATCH --output=result.out
#SBATCH --error=error.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=12:00:00

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

cd "\${SLURM_SUBMIT_DIR}"

lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=1

# Node-local binary copy + startup-flake retry (see production submit.sh).
lmp_local=\${SLURM_TMPDIR:-/tmp}/lmp_\${SLURM_JOB_ID}
if cp "\${lmp}" "\${lmp_local}" 2>/dev/null; then
    chmod +x "\${lmp_local}"
    lmp="\${lmp_local}"
fi

rseed=\$(( (SLURM_JOB_ID * 2654435761) % 2147483647 ))
[ "\${rseed}" -lt 1 ] && rseed=1
rc=1
for attempt in 1 2 3; do
    start=\${SECONDS}
    srun --export=ALL "\${lmp}" \\
        -var rseed \${rseed} \\
        -var ensemble 0 \\
        -var warmup_steps ${warmup_steps} \\
        -var collision_steps ${collision_steps} \\
        -var anneal_steps ${anneal_steps} \\
        -in simulate.lmp
    rc=\$?
    [ "\${rc}" -eq 0 ] && break
    if [ \$((SECONDS - start)) -gt 600 ]; then
        echo "srun failed rc=\${rc} after >10 min -- real failure" >&2
        break
    fi
    echo "attempt \${attempt} died in \$((SECONDS - start))s -- flake, retrying" >&2
    sleep 20
done
rm -f "\${lmp_local}" 2>/dev/null
exit \${rc}
EOF
        chmod +x "${workdir}/submit.sh"

        ( cd "${workdir}" && sbatch submit.sh )
    done
done

echo ""
echo "Submitted 6 validation jobs (sn @ 300 K, pb @ 0 K; orientations"
echo "100/110/111 each covered).  Monitor with:"
echo "  squeue -u \$USER | grep bd-test"
echo ""
echo "When done, each scratch dir should hold: params.json,"
echo "collision-ion.lammpstraj, anneal-ion.lammpstraj, final.data (~180-250 MB)."