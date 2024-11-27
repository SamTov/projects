#/bin/bash


gmx_mpi msd -f traj_comp.xtc -s mdrun.tpr -b 500 -endfit 40000 -o msd.xvg -mol water.xvg <<< 1 # Diffusion
gmx_mpi rdf -f traj_comp.xtc -s mdrun.tpr -b 1000 -ref 1 -sel 1 -selrpos whole_mol_com -seltype whole_mol_com -o rdf.xvg  # RDF
gmx_mpi hbond -f traj_comp.xtc -s mdrun.tpr -b 500 -life -ac  <<< '1 1' # H bond stats
