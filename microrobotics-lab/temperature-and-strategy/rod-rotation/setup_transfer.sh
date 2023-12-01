# Script for making the directory setup of the
# RL experiments.

declare -A hashmap

hashmap[0]=18
hashmap[150]=3
hashmap[273]=16
hashmap[300]=10

script=rod-rotation-deployment.py
for i in 0 150 273 300 350
do
	mkdir ${i}K/transfer
    cp ${script} ${i}K/transfer
    sed -i "s/TEMP/${i}./g" ${i}K/transfer/${script}

    for j in 0 150 273 300
    do
        mkdir ${i}K/transfer/${j}K
        cp ${i}K/transfer/${script} ${i}K/transfer/${j}K/
        cp -r ${j}K/${hashmap[$j]}/Models ${i}K/transfer/${j}K/
    done
	 
done
