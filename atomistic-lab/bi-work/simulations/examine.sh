#!/bin/bash

# Array of directories
directories=("100_molecules" "500_molecules" "1000_molecules" "1500_molecules" "2000_molecules" "2500_molecules" "3000_molecules" "3500_molecules" "4000_molecules")

# Loop over each directory
for dir in "${directories[@]}"; do
    echo "Submitting job for directory: $dir"
    
    # Change into the directory
    cd "$dir"
    
    # Call sbatch submit.sh
    sbatch submit_analysis.sh
    
    # Go back to the parent directory
    cd ..
done

echo "All jobs submitted."
