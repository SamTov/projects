#!/bin/bash

# Arrays of parameter values
coupling_strengths=(0.0 0.01 0.1 0.5 1.0 2.0 5.0 10.0 50.0 100.0 500.0 1000.0)
state_sizes=(1 10 20 50 100 500 1000)
prediction_lengths=(1 10 20 50 100 500)

# Loop over each parameter
for coupling in "${coupling_strengths[@]}"; do
    for state_size in "${state_sizes[@]}"; do
        for prediction_length in "${prediction_lengths[@]}"; do
            # Submit the job with sbatch
            sbatch submit.sh "$coupling" "$state_size" "$prediction_length"
        done
    done
done
