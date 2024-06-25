#! /bin/bash

# qRC Study

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH --job-name=muP  # Job name
#SBATCH --output=result.out              # Output file
#SBATCH --error=error.err                # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --cpus-per-task=12                # Number of CPU cores per task
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00                  # Time limit hrs:min:sec

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

source ~/.bashrc  # Provide access to conda
conda activate nanopore  # activate your conda environment

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

# Run the script with arguments.
python main.py --arch resnet18 --width_mult 2 --optimizer muadam --load_base_shape resnet18.bsh --batch_size 24 --accumulate_batches 500 --num_workers 8 --resume
