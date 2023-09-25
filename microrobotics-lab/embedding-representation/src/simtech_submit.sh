#! /bin/bash

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J VisionEmbedding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --partition cpu

### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stovey@icp.uni-stuttgart.de

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load gcc/11.3.0
module load openmpi/4.1.4_gcc-11.3_cuda-11.7
module load cmake/3.26.3
module load boost/1.82.0_gcc-11.7_ompi-4.1.4

source ~/.bashrc

pypresso=/data/work/ac134186/repositories/espresso/build/pypresso
script=SCRIPT

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

export OMP_NUM_THREADS=${SLURM_NTASKS}

${pypresso} ${script} > output.out &
