#/usr/bin/bash

# ************************************************************ #
# Decision boundary network evolution experiment and analysis. #
#                                                              #
# Author: Samuel Tovey                                         #
# Contact: stovey@icp.uni-stuttgart.de                         #
# ************************************************************ #

# Activate the correct environment
conda activate zincware

# Directories in which to perform studies
directories=( single-output double-output double-output-ce )

# Scripts important to the analysis and experiment.
experimentScript=experiment.py
analysisScript=analysis-script.py
analysisFunctions=analysis_functions.py

for directory in ${directories};
do
	# Copy files to the experiment directory
	cp ${analysisScript} ${directory}
	cp ${analysisFunctions} ${directory}
	
	# Chnage into experiment directory
	cd ${directory}

	# Run the network training
	python ${experimentScript}

	# Run the analysis code
	python ${analysisScript}

	# Change back to home directory
	cd ../
done

