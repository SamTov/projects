#! /bin/bash

# RL Brownian Studies

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J temperature-study

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition single
#SBATCH --time=48:00:00
#SBATCH --mem=24gb

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###


module load compiler/gnu/12.1   
module load mpi/openmpi/4.1
module load devel/cmake/3.24.1


source ~/.bashrc

pypresso=/home/st/st_st/st_ac134186/software/SwarmRL/espresso/build/pypresso
script=rod-rotation-rl.py
# script=rod-rotation-deploy.py


### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

${pypresso} ${script} >> output.out
