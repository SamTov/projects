export LC_NUMERIC="en_US.UTF-8"

sizes=($(seq 0.025 0.075 2.5))
speeds=($(seq 1 0.5 5))
temperatures=( 300 ) # ($(seq 150 50 400))

home=$(pwd)

# Loop over temperature
for t in ${temperatures[@]}
do
        for s in ${speeds[@]}
        do
                for r in ${sizes[@]}
                do
                        directory=${t}/${s}/${r}mum
			cd ${directory}/${i}
			sbatch submit.sh
			cd ${home}
                done
        done
done
