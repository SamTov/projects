"""
Module for selecting loss functions.
"""
import znnl as nl
import optax


def select_loss_fn(name: str):
    """
    Select a loss function.
    """
    if name is "ce":
        loss_fn = nl.loss_functions.CrossEntropyLoss()
    elif name is "mse":
        loss_fn = nl.loss_functions.MeanPowerLoss(order=2)
    elif name is "bce":
        loss_fn = optax.sigmoid_binary_cross_entropy

    return loss_fn
