"""
Helper functions for sleecting activation functions.
"""
import flax.linen as nn


def select_activation(name: str):
    """
    Select an optimizer.
    """
    if name is "relu":
        activation = nn.relu
    elif name is "sigmoid":
        activation = nn.sigmoid
    elif name is "tanh":
        activation = nn.tanh
    else:
        raise ValueError("Activation is not supported.s")
    
    return activation
