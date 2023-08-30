# Script for making the directory setup of the
# RL experiments.

export LC_NUMERIC="en_US.UTF-8"

script=find-center-rl.py

sizes=($(seq 0.025 0.075 2.5))
speeds=($(seq 1 0.5 5))
temperatures=( 300 ) # ($(seq 50 50 400))
ensembles=($(seq 1 1 20))

# Loop over temperature
for t in ${temperatures[@]}
do
	for s in ${speeds[@]}
	do
		for r in ${sizes[@]}
		do
			directory=${t}/${s}/${r}mum
			mkdir -p ${directory}
			cp ${script} ${directory}
			cp submit.sh ${directory}
			sed -i "s/TEMPERATURE/${t}/g" ${directory}/${script}
			sed -i "s/SPEED/${s}/g" ${directory}/${script}
			sed -i "s/RADIUS/${r}/g" ${directory}/${script}

			for i in ${ensembles[@]}
			do
				mkdir ${directory}/${i}
				cp ${directory}/${script} ${directory}/${i}
			done
		done
	done
done

