#!/usr/bin/ruby


# Make a new directory.
def make_directory(directory)
  mkdir_cmd = "mkdir " + directory
  system(mkdir_cmd)

end


# Copy files from reference dir to new dir.
def copy_files(directory)
  # Define the commands
  cp_cmd_1 = "cp templates/analysis.sh " + directory
  cp_cmd_2 = "cp templates/submit_analysis.sh " + directory
  
  # Run the commands
  system(cp_cmd_1)
  system(cp_cmd_2)
end


# Main run method.
def main()
  n_atoms = Array[100, 110, 120, 130, 150]
  for item in n_atoms
    directory = item.to_s + "_molecules"  # create directory name

    copy_files(directory)  # copy the files to the new directory
    
  end

end


main()
