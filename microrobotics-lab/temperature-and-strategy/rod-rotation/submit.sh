#! /bin/bash

# RL Brownian Studies

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J temperature-study

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --partition cpu


### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stovey@icp.uni-stuttgart.de

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load gcc/11.3.0
module load openmpi/4.1.4_gcc-11.3_cuda-11.7
module load cmake/3.26.3
module load boost/1.82.0_gcc-11.7_ompi-4.1.4

source ~/.bashrc

pypresso=/data/work/ac134186/repositories/SwarmRL/espresso/build/pypresso
script=rod-rotation-rl.py

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

for i in 1 2 3 4 5 6 7 8 9 10
do
	cd ${i}
	OMP_NUM_THREADS=12 ${pypresso} ${script} >> output.out &
	cd ../
done
wait
