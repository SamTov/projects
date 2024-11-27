#!/usr/bin/ruby


# Make a new directory.
def make_directory(directory)
  mkdir_cmd = "mkdir " + directory
  system(mkdir_cmd)

end


# Copy files from reference dir to new dir.
def copy_files(directory)
  # Define the commands
  cp_cmd_1 = "cp templates/water.top " + directory
  cp_cmd_2 = "cp templates/mdout.mdp " + directory
  cp_cmd_3 = "cp templates/submit.sh " + directory
  cp_cmd_4 = "cp templates/restart.sh " + directory
  
  # Run the commands
  system(cp_cmd_1)
  system(cp_cmd_2)
  system(cp_cmd_3)
  system(cp_cmd_4)
end


# Edit the topology file. You must be in the directory of the file.
def edit_topology(n_atoms, directory)
  edit_cmd = "sed -i  's/NATOMS/" + n_atoms.to_s + "/g' " + directory + \
    "/water.top"

  system(edit_cmd)
end


# Edit the slurm submit script
def edit_submit(n_atoms, directory)
  edit_cmd = "sed -i 's/NMOLS/" + n_atoms.to_s + "/g' " + directory + "/submit.sh"

  system(edit_cmd)
end



# Run the solvate command to fill the box.
def run_solvate(n_atoms, directory)
  solvate_run = "/group/allatom/gromacs-4.6.7/bin/genbox -cs spc216 -o " + directory + "/run" + n_atoms.to_s + ".gro -box 5.0 5.0 5.0 -try 500 -maxsol " + n_atoms.to_s
  system(solvate_run)

end


# Main run method.
def main()
  n_atoms = Array[100, 110, 120, 130, 150, 200, 250, 300]
  for item in n_atoms
    directory = item.to_s + "_molecules"  # create directory name

    make_directory(directory)  # create the directory
    copy_files(directory)  # copy the files to the new directory
    edit_submit(item, directory)  # edit the slurm file
    edit_topology(item, directory)  # edit the topology
    run_solvate(item, directory)  # build the simulation box
    
  end

end

main()

