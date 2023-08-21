# Script for making the directory setup of the
# RL experiments.

export LC_NUMERIC="en_US.UTF-8"

script=embedding-study.py

embeddings=($(seq 1 1 5))
ensembles=($(seq 1 1 20))

# Loop over temperature
for e in ${embeddings[@]}
do
	directory=${e}_dimensions
	mkdir -p ${directory}
	cp ${script} ${directory}
	cp submit.sh ${directory}
	sed -i "s/KNOWLEDGE/${e}/g" ${directory}/${script}

	for i in ${ensembles[@]}
	do
		mkdir ${directory}/${i}
		rm -r ${directory}/${i}/deployment
		cp ${directory}/${script} ${directory}/${i}
	done

done

