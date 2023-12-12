#! /bin/bash

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J InitStudy

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition single
#SBATCH --time=2:00:00
#SBATCH --mem=55gb
#SBATCH --gres=gpu:1
#SBATCH -e slurm_files/slurm-%j.err
#SBATCH -o slurm_files//slurm-%j.out

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###


module load compiler/gnu/12.1   
module load mpi/openmpi/4.1
module load devel/cuda/12.1 
module load devel/cmake/3.24.1
module load devel/cuda/12.1

source ~/.bashrc
conda activate zincware

script=py_submit.py

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python ${script} >> output.out