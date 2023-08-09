#! /bin/bash

# RL Brownian Studies

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J TASKNAME

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stovey@icp.uni-stuttgart.de

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load gcc/8.4.0
module load openmpi/4.0.5
module load cmake/3.19.2
module load fftw/3.3.9
module load boost/1.75.0

pypresso=/beegfs/work/stovey/work/PhD/microrobots-brownian/espresso/build/pypresso
#script=find-center-rl.py
script=find-center-deployment.py

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

${pypresso} ${script} > output.out

