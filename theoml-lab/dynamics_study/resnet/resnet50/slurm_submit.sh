#! /bin/bash

# TheoML Dynamics Study

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J resnet18-dynamics

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition single
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=32gb
#SBATCH --output=result.out
#SBATCH --error=error.err

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load compiler/gnu/12.1   
module load mpi/openmpi/4.1
module load devel/cmake/3.24.1

# Source the bashrc file.
source ~/.bashrc

# Activate the conda environment.
conda activate zincware

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd ${SLURM_SUBMIT_DIR}  # change into working directory
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python train.py
