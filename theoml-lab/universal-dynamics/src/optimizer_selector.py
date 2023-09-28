"""
Module for selecting optimizers.
"""
import optax

def select_optimizer(name: str):
    """
    Select an optimizer.
    """
    if name is "adam":
        optimizer = optax.adam
    elif name is "sgd":
        optimizer = optax.sgd
    else:
        raise ValueError("Optimizer is not supported.")
    
    return optimizer
