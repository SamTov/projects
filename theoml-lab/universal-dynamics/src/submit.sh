#! /bin/bash

# Universal dynamics study

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J nn-dynamics

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition single
#SBATCH --time=48:00:00
#SBATCH --mem=42gb
#SBATCH --gres=gpu:1

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load compiler/gnu/12.1   
module load mpi/openmpi/4.1
module load devel/cuda/12.1 
module load devel/cmake/3.24.1

source ~/.bashrc
conda activate zincware

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
export OMP_NUM_THREADS=${SLURM_NTASKS}

python experiment.py >> output.out
