#! /bin/bash

# Ballistic Diamond

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH --job-name=ballistic-diamond     # Job name
#SBATCH --output=result.out              # Output file
#SBATCH --error=error.err                # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=64                # Number of task
#SBATCH --time=4:00:00                  # Time limit hrs:min:sec

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load spack/default
module load gcc/12.3.0 
module load openmpi/4.1.6
module load fftw/3.3.10

source ~/.bashrc  # Provide access to conda
conda activate nanopore  # activate your conda environment

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
lmp=/home/stovey/work/projects/quantum-lab/ballistic-diamond/lammps/build/lmp
export OMP_NUM_THREADS=32
# Run the script with arguments.
srun ${lmp} -var rseed ${RANDOM} -in simulate.lmp