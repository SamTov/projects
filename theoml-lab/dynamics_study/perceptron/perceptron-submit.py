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
    

dataset = "linear"
ds_size=200 
epochs=500
lr=1.0
batch_size=32
architecture='perceptron'
widths=[1]
depths=[1]
activations=["relu", "tanh", "sigmoid"]
input_shape="1 2"
accuracy=True
loss_fn="cross_entropy"
ntk_batch=100

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
