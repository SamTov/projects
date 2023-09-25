# Makefile for deploying experiment on either the SimTech or Helix
# Clusters. 
# The following parameters are present:
#   * cluster: Which cluster you are on.
#	* mode: Either training or deployment.

cluster=helix
mode=training

.SILENT:

setup:
	make ${cluster}_setup
	make clean_setup

deploy:
	cp src/recursive_submit.sh .
	bash recursive_submit.sh
	make clean_deploy

helix_setup:
	cp src/helix_submit.sh src/${mode}.py src/${mode}_setup.sh .
	bash ${mode}_setup.sh helix_submit.sh

simtech_setup:
	cp src/simtech_submit.sh src/${mode}.py src/${mode}_setup.sh .
	bash ${mode}_setup.sh simtech_submit.sh

clean_setup:
	rm ${cluster}_submit.sh ${mode}.py ${mode}_setup.sh

clean_deploy:
	rm recursive_submit.sh

purge:
	rm -rf *_dimensions
