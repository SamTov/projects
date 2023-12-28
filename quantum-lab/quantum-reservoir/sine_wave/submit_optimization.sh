#!/bin/bash

# Define the ranges of the parameters
coupling_values=$(seq 0.01 0.01 0.5)
dimension_values=$(seq 5 5 5000)
relaxation_values=$(seq 1 1 1000)

# Iterate over all combinations of parameters
for coupling in $coupling_values; do
    for dimension in $dimension_values; do
        for relaxation in $relaxation_values; do
            # Backup original experiment.py
            cp experiment.py experiment.py.bak

            # Modify experiment.py with the current set of parameters
            sed -i "s/coupling=.*,/coupling=$coupling,/" experiment.py
            sed -i "s/state_dimension=.*,/state_dimension=$dimension,/" experiment.py
            sed -i "s/relaxation_time=.*,/relaxation_time=$relaxation)/" experiment.py

            # Submit the job to SLURM
            sbatch run_experiment.slurm
            
            # Restore the original experiment.py
            mv experiment.py.bak experiment.py
        done
    done
done
