#!/bin/bash

# Array of directories
directories=("100_molecules" "110_molecules" "120_molecules" "130_molecules" "150_molecules")

# Loop over each directory
for dir in "${directories[@]}"; do
    echo "Submitting job for directory: $dir"
    
    # Change into the directory
    cd "$dir"
    
    # Call sbatch submit.sh
    sbatch restart.sh
    
    # Go back to the parent directory
    cd ..
done

echo "All jobs submitted."
