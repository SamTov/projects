# Script for making the directory setup of the
# RL experiments.

export LC_NUMERIC="en_US.UTF-8"

script=embedding-study.py

embeddings=($(seq 1 1 5))

# Loop over temperature
for e in ${embeddings[@]}
do
	directory=${e}_dimensions
	cd ${directory}
	sbatch submit.sh
	cd ../
done

