#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# Large scale packages
import znnl as nl

# Linalg help
import numpy as np
# import jax.numpy as np
from flax.training import train_state
import flax.linen as nn
import optax
import jax
from flax.core.frozen_dict import FrozenDict
from jax.tree_util import tree_flatten, tree_unflatten

# system and plotting help
import matplotlib.pyplot as plt
from dataclasses import dataclass
import copy
from rich.progress import track


# ## Model Definition

# In[ ]:


class DenseNet(nn.Module):
    """
    Simple CNN module.
    
    Parameters
    ----------
    w_std : float
            Weight std for initalization.
    b_std : float
            Bias std for initialization.
    """
    w_std : float
    b_std : float
    
    def setup(self):
        """
        Create the initializers.
        """
        self.kernel_init = nn.initializers.normal(
            self.w_std
        )
        self.bias_init = nn.initializers.normal(
            self.b_std
        )

    @nn.compact
    def __call__(self, x):
        """
        Forward propogation through the network.
        """
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            features=128, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            features=10,
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init
        )(x)

        return x


# ## Experiment Helpers

# In[ ]:


@dataclass
class Experiment:
    # Experiment Parameters
    loss_fn: str
    width: int
    depth: int
    learning_rate: float
    optimizer: str
    epochs: int
    batch_size: int
    ds_size: int
    
    # Results
    parameters: list
    train_loss: np.ndarray


# In[ ]:


def main(
    w_std: float,
    b_std: float,
    learning_rate: float = 1e-2,
    batch_size: int = 128,
    epochs: int = 100,
    generator: callable = nl.data.MNISTGenerator(500) 
):
    """
    Run the experiment.
    """
    prefix="./" # "/data/stovey/initialization"
    
    experiment_results = []
    
    # Create the model
    network = DenseNet(w_std=w_std, b_std=b_std)
    
    model = nl.models.FlaxModel(
            flax_module=network,
            optimizer=optax.adam(learning_rate=learning_rate),
            batch_size=100,
            input_shape=(1, 28, 28, 1),
    )
    
    # Recorders
    train_recorder = nl.training_recording.JaxRecorder(
        name=f"{prefix}/{w_std}_{b_std}_train_recorder",
        loss=True,
        accuracy=True,
        update_rate=1
    )
    test_recorder = nl.training_recording.JaxRecorder(
        name=f"{prefix}/{w_std}_{b_std}_test_recorder",
        loss=True,
        accuracy=True,
        update_rate=1
    )
    cv_recorder = nl.training_recording.JaxRecorder(
        name=f"{prefix}/{w_std}_{b_std}_cv_recorder",
        entropy=True,
        trace=True,
        update_rate=1000
    )
    
    train_recorder.instantiate_recorder(
        data_set=generator.train_ds
    )
    test_recorder.instantiate_recorder(
        data_set=generator.test_ds
    )
    cv_recorder.instantiate_recorder(
        data_set=generator.train_ds
    )
    
    # Training Strategy.
    training_strategy = nl.training_strategies.SimpleTraining(
        model=model, 
        loss_fn=nl.loss_functions.MeanPowerLoss(order=2),
        accuracy_fn=nl.accuracy_functions.LabelAccuracy(),
        recorders=[train_recorder, test_recorder, cv_recorder]
    )
    
    # Train the model.
    _ = training_strategy.train_model(
        train_ds=generator.train_ds,
        test_ds=generator.test_ds,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Dump the recorders.
    train_recorder.dump_records()
    test_recorder.dump_records()
    cv_recorder.dump_records()           

