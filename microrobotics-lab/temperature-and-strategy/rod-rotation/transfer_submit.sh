# Submit jobs recursively

for i in 0 150 273 300 350
do
    for j in 0 150 273 300
    do
        cp submit.sh ${i}K/transfer/${j}K/
        cd ${i}K/transfer/${j}K/
        sbatch submit.sh
        cd ../../..
    done
	
done
