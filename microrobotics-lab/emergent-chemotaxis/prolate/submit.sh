#! /bin/bash

# RL Brownian Studies

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J EmbeddingStudy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --partition cpu,cpu-long


### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
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
script=find-center-rl.py

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory

ensembles=($(seq 1 1 20))

for i in ${ensembles[@]}
do
	# Change into ensemble directory.
	cd ${i}
	# Run the script on specific number of nodes
	OMP_NUM_THREADS=6 ${pypresso} ${script} > output.out &
	# Change back out
	cd ../
done
wait

