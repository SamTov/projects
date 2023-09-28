#! /usr/bin/ruby

# Set default parameters
ds_size = 1000
dataset = "\"MNIST\""
one_hot = "False"
learning_rate = 0.02
input_shape = "(1, 28, 28, 1)"
accuracy = "True"
batch_size = 256
epochs = 1000
loss_fn = "\"mse\""
activation = "\"relu\""
use_bias = "True"

# number of ensembles
n_ensembles = 20

sed_hash = {
    "DSSIZE" => ds_size, 
    "DATASET" => dataset, 
    "ONEHOT" => one_hot, 
    "LR" => learning_rate, 
    "INPUT" => input_shape, 
    "ACCURACY" => accuracy, 
    "BATCH" => batch_size, 
    "EPOCHS" => epochs, 
    "LOSS" => loss_fn, 
    "BIAS" => use_bias,
    "ACTIVATION" => activation,
}

# Define parameters to be tested
output_dimensions = Array.[](10)
network_depths = Array.[](1, 2, 3)
network_widths = Array.[](1, 2, 10, 100)
optimizers = Array.[]("\"adam\"", "\"sgd\"")

parameter_hash = {
    "output_dimensions" => output_dimensions,
    "network_depths" => network_depths,
    "network_widths" => network_widths,
    "optimizers" => optimizers
}

# Define helper functions

# Applies the sed operation on the experiment file.
#
# @param filename [String] file to edit.
# @param old_value [String] name of the parameter to change.
# @param new_value [String] what to change the old parameter into.
def sed_op(filename, old_value, new_value)
    sed_cmd = "sed -i 's/" + old_value.to_s + "/" + new_value.to_s + "/g' " + filename.to_s
    system(sed_cmd)
end

# Set the fixed default parameters of the study.
#
# @param filename [String] which file to operate on.
# @param defaults_hash [Hash] default parameters to use.
def set_defaults(filename, defaults_hash)

    # Loop over the hash and perform the sed operation.
    defaults_hash.each do |key, value|
        sed_op(filename, key, value)
    end

end

# Prepare the directory structure

# Builds the directory structure for the experiment.
#
# @param
# @param
# @param
def setup(default_parameters, loop_parameters, n_ensembles)
    # Create the experiment directory
    root = "mnist-dense"
    system("mkdir " + root.to_s)

    # copy experiment template into root directory
    system("cp src/dense_experiment.py " + root.to_s + "/experiment.py")
    experiment_root = root + "/experiment.py"

    # Set default parameters of experiment
    set_defaults(experiment_root, default_parameters)

    # Loop and create
    for network_depth in loop_parameters["network_depths"]
        for network_width in loop_parameters["network_widths"]
            for output_dimension in loop_parameters["output_dimensions"]
                for optimizer in loop_parameters["optimizers"]
                    # Create the directory.
                    creation_dir = root + "/" + network_depth.to_s + "/" 
                    creation_dir += network_width.to_s + "/" 
                    creation_dir += output_dimension.to_s + "/" 
                    creation_dir += optimizer + "/"

                    # Create dir and copy experiment file.
                    system("mkdir -p " + creation_dir)  # Create the directory
                    system("cp " + experiment_root + " " + creation_dir)
                    experiment_subroot = creation_dir + "experiment.py"

                    # Change experiment parameters
                    sed_hash = {
                        "DEPTH": network_depth,
                        "WIDTH": network_width,
                        "OUTPUT": output_dimension,
                        "OPTIMIZER": optimizer,
                    }

                    sed_hash.each do |key, value|
                        sed_op(experiment_subroot, key, value)
                    end

                    # Create ensembles and copy over experiment scripts
                    ensembles = Array(1..n_ensembles.to_int)
                    for ensemble in ensembles
                        ensemble_path = creation_dir.to_s + ensemble.to_s
                        system("mkdir " + ensemble_path.to_s)
                        system("cp " + experiment_subroot + " " + ensemble_path)
                    end
                end
            end
        end
    end
end

# Deploy the study

# Deploys experiments on the cluster.
#
# @param loop_parameters [Mesh] Parameters used in the directory build.
# @param n_ensembles [int] Number of ensembles being performed.
def deploy(loop_parameters, n_ensembles)
    # Create the experiment directory
    root = "mnist-dense"
    experiment_root = root + "/experiment.py"

    # Loop and create
    for network_depth in loop_parameters["network_depths"]
        for network_width in loop_parameters["network_widths"]
            for output_dimension in loop_parameters["output_dimensions"]
                for optimizer in loop_parameters["optimizers"]
                    creation_dir = root + "/" + network_depth.to_s + "/" 
                    creation_dir += network_width.to_s + "/" 
                    creation_dir += output_dimension.to_s + "/" 
                    creation_dir += optimizer + "/"
                    
                    # Submit the jobs
                    ensembles = Array(1..n_ensembles.to_int)
                    for ensemble in ensembles
                        ensemble_path = creation_dir.to_s + ensemble.to_s

                        # Copy submit file into the ensemble directory
                        system("cp src/submit.sh "  + ensemble_path)
                        system("sbatch " + ensemble_path.to_s + "/submit.sh")
                    end
                end
            end
        end
    end
end

if ARGV[0] == "setup"
    setup(sed_hash, parameter_hash, n_ensembles)
elsif ARGV[0] == "deploy"
    deploy(parameter_hash, n_ensembles)
else
    raise StandardError.new "Invalid experiment option."
end
