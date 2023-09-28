"""
Helper module to build datasets from strings.
"""
import znnl as nl

from znnl.data.decision_boundary import (
    DecisionBoundaryGenerator,
    circle,
    linear_boundary,
)

def build_generator(alias: str, ds_size: int, one_hot: bool = False):
    """
    Create a data generator from parameters.

    Parameters
    ----------
    alias : str
        Name of the dataset to build.
    ds_size : int
        Size of the dataset.
    one_hot : bool (default = False)
        If true, one-hot encode the targets.
    """
    if alias == "line":
        generator = DecisionBoundaryGenerator(ds_size, discriminator="line")
    elif alias == "circle":
        generator = DecisionBoundaryGenerator(ds_size, discriminator="circle")
    elif alias == "MNIST":
        generator = nl.data.MNISTGenerator(ds_size, one_hot)
    elif alias == "CIFAR10":
        generator = nl.data.CIFAR10Generator(ds_size, one_hot)
    elif alias == "fuel":
        generator = nl.data.MPGDataGenerator(train_fraction=0.8)
        generator.train_ds["inputs"] = generator.train_ds["inputs"][:300]
        generator.train_ds["targets"] = generator.train_ds["targets"][:300]
    elif alias == "abalone":
        generator = nl.data.AbaloneDataGenerator(train_fraction=0.8)
        generator.train_ds["inputs"] = generator.train_ds["inputs"][:3300]
        generator.train_ds["targets"] = generator.train_ds["targets"][:3300]

    return generator
