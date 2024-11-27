# Process management
import subprocess as sp

# Linalg support
import numpy as np

# Define scan parameters
coupling_strengths = np.unique(np.logspace(0, 2, 20, dtype=int))
state_sizes = np.unique(np.logspace(0, 2, 20, dtype=int))
# prediction_lengths = (1, 10, 20, 50, 100)
prediction_lengths = (500,)

for coupling_strength in coupling_strengths:
    for state_size in state_sizes:
        for prediction_length in prediction_lengths:
            # Run the scan
            sp.Popen(["sbatch", "submit.sh", str(coupling_strength), str(state_size), str(prediction_length)])
