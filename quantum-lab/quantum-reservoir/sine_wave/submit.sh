#! /bin/bash

# qRC Study

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH --job-name=qRC_parameter_search  # Job name
#SBATCH --output=result.out              # Output file
#SBATCH --error=error.err                # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec
#SBATCH --exclusive                      # Exclusive use of node

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

source ~/.bashrc
conda activate zincware

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

python crop_code.py > output.out