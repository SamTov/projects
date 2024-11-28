#! /bin/bash

# Water simulation

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J NMOLS_Water_simulation

#SBATCH -n 32
#SBATCH --exclude=compute-big-0-0,compute-big-0-1,compute-quadgpu-1-0

### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stovey@icp.uni-stuttgart.de

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load gcc/8.4.0
module load openmpi/4.0.5
module load gromacs/2020.4


### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

gmx_mpi mdrun -cpi state.cpt -s mdrun.tpr 
