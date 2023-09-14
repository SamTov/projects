# Submit jobs recursively

for i in 0 150 273 300 350
do
	cp submit.sh ${i}K
	cd ${i}K
	sbatch submit.sh
	cd ../
done
