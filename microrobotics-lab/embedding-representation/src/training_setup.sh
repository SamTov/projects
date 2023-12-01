# Script for making the directory setup of the
# RL experiments.

export LC_NUMERIC="en_US.UTF-8"

script=training.py  # name of script to deploy
submit_script=$1

embeddings=($(seq 1 2 10))
ensembles=($(seq 1 1 20))

# Loop over temperature
for e in ${embeddings[@]}
do
	for i in ${ensembles[@]}
	do
        # Create the new directories.
        directory=${e}_dimensions/${i}
        mkdir -p ${directory}

        # Copy files into directory.
        cp ${script} ${directory}
        cp ${submit_script} ${directory}

        # Update the parameters in the files.
        sed -i "s/KNOWLEDGE/${e}/g" ${directory}/${script}
        sed -i "s/SCRIPT/${script}/g" ${directory}/${submit_script}

	done

done

