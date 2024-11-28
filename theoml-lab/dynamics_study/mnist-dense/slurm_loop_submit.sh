#! /bin/bash

# TheoML Dynamics Study

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J mnist-dynamics

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition single
#SBATCH --gres=gpu:1
#SBATCH --time=110:00:00
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


# python train.py --data ${1} --size ${2} --epochs ${3} --lr ${4} --batch ${5} --architecture ${6} --width ${7} --depth ${8} --activation ${9} --ntk_batch ${10} --input_shape ${11} --accuracy ${12} --loss_fn ${13}
python loop_submit.py