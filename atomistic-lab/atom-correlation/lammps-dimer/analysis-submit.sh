#! /bin/bash


### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH --job-name=water-analysis  # Job name
#SBATCH --output=result.out              # Output file
#SBATCH --error=error.err                # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

source ~/.bashrc  # Provide access to conda
conda activate atomistic  # activate your conda environment

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

# Run the script with arguments.
python analysis.py
