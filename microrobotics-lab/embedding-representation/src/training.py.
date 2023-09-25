#! /bin/bash

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J VisionEmbedding

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition single
#SBATCH --time=48:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###


module load compiler/gnu/12.1   
module load mpi/openmpi/4.1
module load devel/cuda/12.1 
module load devel/cmake/3.24.1


source ~/.bashrc

pypresso=/home/st/st_st/st_ac134186/software/SwarmRL/espresso/build/pypresso
script=SCRIPT

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
export OMP_NUM_THREADS=${SLURM_NTASKS}

${pypresso} ${script} >> output.out
