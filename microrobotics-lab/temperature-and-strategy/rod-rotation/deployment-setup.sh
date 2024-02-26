# Script for making the directory setup of the
# RL experiments.

script=rod-rotation-deploy.py
for i in 0 150 273 300 350
do
	cp ${script} ${i}K
	cd ${i}K
	sed -i "s/TEMP/${i}./g" ${script}
	for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
	do
		cp ${script} $j
        rm -rf ${j}/deployment
	done
	cd ../
done
