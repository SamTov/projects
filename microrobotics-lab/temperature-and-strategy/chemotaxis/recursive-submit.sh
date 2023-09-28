# Submit jobs recursively

for i in 0 150 273 300 350
do
	cp submit.sh ${i}K
	for j in 1 2 3 4 5 6 7 8 9 10
	do
		cd ${i}K
		cp submit.sh ${j}
		cd ${j}
		sbatch submit.sh
		cd ../../
	done
done
