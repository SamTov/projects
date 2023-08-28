#/bin/bash
# ------------------------------ #
# Perform LJ MD simulation study #
# ------------------------------ #

run_script=lj.lmp

# Gas phase simulations
for t in $(seq 0.001 0.001 0.01);
do
	dir_name=lj_${t}
	mkdir ${dir_name}
	cp ${run_script} ${dir_name}
	cd ${dir_name}
	gsed -i "s|PRESS|${t}|g" ${run_script}
	mpirun -np 4 lmp_mpi -in ${run_script}
	cd ../
done

# Liquid phase simulations
for t in $(seq 0.05 0.005 0.5);
do
        dir_name=lj_${t}
        mkdir ${dir_name}
        cp ${run_script} ${dir_name}
        cd ${dir_name}
        gsed -i "s|PRESS|${t}|g" ${run_script}
        mpirun -np 4 lmp_mpi -in ${run_script}
        cd ../
done
