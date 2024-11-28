"""
Module for running the fuel ds experiments.
"""
# ZnNL
import znnl as nl

# Helper libaries
from dataclasses import dataclass
import subprocess as sp
import numpy as onp

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

dataset = "mnist"
ds_size=1000  # for fuel this is a train fraction.
epochs=500
lr=0.001
batch_size=128
architecture='dense'
# widths=[4, 12, 50, 100, 500, 1000]
widths = list(onp.unique(onp.logspace(1, 3.5, 20, dtype=int)))
# depths=[1, 2, 3, 5, 6, 7, 8, 9, 10]
depths=[4]

activations=["relu", "tanh"]
input_shape="1 28 28 1"
accuracy=True
loss_fn="cross_entropy"
ntk_batch=50

for activation in activations:
    for width in widths:
        for depth in depths:
            for _ in range(2):
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