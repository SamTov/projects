# Submit jobs recursively

ensembles=($(seq 1 1 20))

for i in 0 150 273 300 350
do
	cp submit.sh ${i}K
	cd ${i}K
	for j in ${ensembles[@]}
	do
		cp submit.sh ${j}
		cd ${j}
		rm -rf ep_training
		sbatch submit.sh
		cd ../
	done
	cd ../
done
