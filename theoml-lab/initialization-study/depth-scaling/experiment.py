# # Initialization Study

import znnl as nl

# ML modules
import flax.linen as nn
import optax

# Linalg modules
import numpy as np

# System modules
import time
from dataclasses import dataclass


class DenseNetwork(nn.Module):
        """
        Dense network architecture.

        Attributes
        ----------
        width : int
            Width of the network.
        depth : int
            Depth of the network.
        output_dim : int
            Dimension of the output.
        activation : callable
            Activation function.
        w_std : float
            Standard deviation of the weights.
        b_std : float
            Standard deviation of the biases.
        kernel_init : callable
            Kernel initializer.
        bias_init : callable
            Bias initializer.
        """
        # Architecture parameters
        width: int
        depth: int
        output_dim: int
        activation: str

        # Initialization parameters
        w_std: float
        b_std: float

        def setup(self):
            """
            Setup the network.
            """
            self.kernel_init = nn.initializers.normal(self.w_std)
            self.bias_init = nn.initializers.normal(self.b_std)

        @nn.compact
        def __call__(self, x):
            """
            Call the network.

            Parameters
            ----------
            x : array
                Input array.

            Returns
            -------
            array
                Output array.
            """
            x = x.reshape((x.shape[0], -1))
            for _ in range(self.depth):
                x = nn.Dense(
                    self.width, 
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init
                )(x)
                x = self.activation(x)
            x = nn.Dense(
                self.output_dim, 
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)
            return x


def experiment(
        width: int, 
        depth: int, 
        activation: callable, 
        w_std: float, 
        b_std: float, 
        lr: float = 1e-2,
        output_dimension: int = 10,
        input_shape: tuple = (1, 28, 28, 1),
        batch_size: int = 128, 
        epochs: int = 501, 
        seed: int = np.random.randint(684831),
        generator: nl.data.DataGenerator = nl.data.MNISTGenerator(500),
    ):
    """
    Run an experiment.

    Parameters
    ----------
    width : int
        Width of the network.
    depth : int
        Depth of the network.
    activation : callable
        Activation function.
    w_std : float
        Standard deviation of the weights.
    b_std : float
        Standard deviation of the biases.
    lr : float (default: 1e-2)
        Learning rate.
    output_dimension : int (default: 10)
        Dimension of the output.
    input_shape : tuple (default: (1, 28, 28, 1))
        Shape of the input.
    batch_size : int (default: 128)
        Batch size.
    epochs : int (default: 1000)
        Number of epochs.
    seed : int (default: random)
        Seed for the random number generator.
    generator : DataGenerator (default: MNISTGenerator(500))

    Returns
    -------
    float
        Final loss.
    """
    # Mandatory print
    print(f"Running experiment with {width} width, {depth} depth, {activation.__name__} activation, {w_std} w_std, {b_std} b_std, {lr} lr, {seed} seed.")

    # Setup the network
    network = DenseNetwork(width, depth, output_dimension, activation, w_std, b_std)

    # Setup the optimizer
    optimizer = optax.adam(lr)

    # File name
    name_seed = np.random.randint(97898365)
    prefix="/data/stovey/init_study_small"
    name = f"{prefix}/{w_std}_{b_std}_{width}_{depth}_{activation.__name__}_{name_seed}"

    # Create the ZnNL model
    model = nl.models.FlaxModel(
            flax_module=network,
            optimizer=optimizer,
            batch_size=10,
            seed=seed,
            input_shape=input_shape,
    )

    # Create the ZnNL recorders
    train_recorder = nl.training_recording.JaxRecorder(
        name=f"{name}_train_recorder",
        loss=True,
        accuracy=True,
        update_rate=1
    )
    test_recorder = nl.training_recording.JaxRecorder(
        name=f"{name}_test_recorder",
        loss=True,
        accuracy=True,
        update_rate=1
    )
    cv_recorder = nl.training_recording.JaxRecorder(
        name=f"{name}_cv_recorder",
        entropy=True,
        trace=True,
        update_rate=50
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

@dataclass
class ExperimentParameters:
    """
    Experiment parameters.
    """
    width: int
    depth: int
    activation: callable
    w_std: float
    b_std: float
    lr: float = 1e-2
    output_dimension: int = 10
    input_shape: tuple = (1, 28, 28, 1)
    batch_size: int = 128
    epochs: int = 501
    seed: int = np.random.randint(684831)
    generator: nl.data.DataGenerator = nl.data.MNISTGenerator(200)

experiment_parameters = ExperimentParameters(
    width=WIDTH,
    depth=DEPTH,
    activation=ACTIVATION,
    w_std=W_STD,
    b_std=B_STD,
)

experiment(**experiment_parameters.__dict__)
