# Submit jobs recursively

ensembles=($(seq 1 1 20))
embeddings=($(seq 1 2 10))

for i in ${embeddings[@]}
do
	cd ${i}_dimensions
	for j in ${ensembles[@]}
	do
		cd ${j}
		sbatch helix_submit.sh
        # echo "I would be submitting now"
		cd ../
	done
	cd ../
done
