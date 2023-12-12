#/bin/bash
export LC_NUMERIC="en_US.UTF-8"

# Default parameters
py_script=experiment.py
py_submit=py_submit.py
submit_script=helix_submit.sh

# SED function for all changes to bash script.
sed_function () {
    change_name=$1
    change_value=$2
    output_file=$3

    sed -i "s/${change_name}/${change_value}/g" ${output_file}
}

# Experiment parameters
# w_stds=($(seq 0 0.1 2))
# b_stds=0.05
# depths=($(seq 1 10 100))
# activations=(nn.relu)
# ensembles=(1)
# widths=(512)

# Test params
w_stds=(1.0)
b_stds=0.05
depths=(1 10 100 300)
activations=(nn.relu)
ensembles=(1)
widths=(512)


# Loop over parameters and submit jobs.
for w_std in ${w_stds[@]}
do
    for depth in ${depths[@]}q
    do
        for activation in ${activations[@]}
        do
            for width in ${widths[@]}
            do
                for i in ${ensembles[@]}
                do  
                    # Copy the submit file
                    cp ${py_script} ${py_submit}

                    # Update the parameters in the files.
                    sed_function "WIDTH" ${width} ${py_submit}
                    sed_function "DEPTH" ${depth} ${py_submit}
                    sed_function "ACTIVATION" ${activation} ${py_submit}
                    sed_function "W_STD" ${w_std} ${py_submit}
                    sed_function "B_STD" ${b_stds} ${py_submit}
                    sed_function "ENSEMBLE" ${i} ${py_submit}

                    # Submit the the cluster
                    sbatch ${submit_script}
                done
            done
        done
    done
done