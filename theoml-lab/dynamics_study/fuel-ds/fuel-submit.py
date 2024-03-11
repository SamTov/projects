"""
Module for running the fuel ds experiments.
"""
# ZnNL
import znnl as nl

# Helper libaries
from dataclasses import dataclass
import subprocess as sp

@dataclass
class ExperimentParameters:
    dataset: nl.data.DataGenerator
    ds_size: float
    epochs: int
    lr: float
    batch_size: int
    architecture: str
    width: int
    depth: int
    activation: str
    ntk_batch: int
    input_shape: tuple
    accuracy: bool
    loss_fn: str

def deploy_experiment(parameters: ExperimentParameters):
    """
    Deploy experiment.
    """
    # Run experiment
    

dataset = "fuel"
ds_size=0.765  # for fuel this is a train fraction.
epochs=500
lr=0.001
batch_size=32
architecture='dense'
widths=[1000]
depths=[2, 3]
activations=["relu", "tanh"]
input_shape="1 9"
accuracy=False
loss_fn="mean_power"
ntk_batch=5

for activation in activations:
    for width in widths:
        for depth in depths:
            for _ in range(5):
                # Create experiment
                experiment = ExperimentParameters(
                    dataset=dataset,
                    ds_size=ds_size,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    architecture=architecture,
                    width=width,
                    depth=depth,
                    activation=activation,
                    ntk_batch=ntk_batch,
                    input_shape=input_shape,
                    accuracy=accuracy,
                    loss_fn=loss_fn
                )
                # Run experiment
                sp.Popen(
                    [
                        "sbatch",
                        "slurm_submit.sh",
                        str(experiment.dataset),
                        str(experiment.ds_size),
                        str(experiment.epochs),
                        str(experiment.lr),
                        str(experiment.batch_size),
                        experiment.architecture,
                        str(experiment.width),
                        str(experiment.depth),
                        experiment.activation,
                        str(experiment.ntk_batch),
                        str(experiment.input_shape),
                        str(experiment.accuracy),
                        experiment.loss_fn
                    ]
                )            
