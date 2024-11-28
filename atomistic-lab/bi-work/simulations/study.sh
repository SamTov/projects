#!/bin/bash

# Array of directories
directories=("100_molecules" "110_molecules" "120_molecules" "130_molecules" "150_molecules" "200_molecules" "250_molecules" "300_molecules" )

# Loop over each directory
for dir in "${directories[@]}"; do
    echo "Submitting analysis job for directory: $dir"
    
    # Change into the directory
    cd "$dir"
    
    # Call sbatch submit.sh
    sbatch submit_analysis.sh
    
    # Go back to the parent directory
    cd ..
done

echo "All jobs submitted."
