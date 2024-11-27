"""
Module for training the networks and storing data.
"""

# Linalg libraries
import jax.numpy as jnp
import numpy as onp

# Neural network libraries
import flax.linen as nn
import optax
from networks import (
    Perceptron,
    DenseNetwork,
    ConvolutionalNetwork
)

# ZnNL
import znnl as nl

# Helper libraries
from argparse import ArgumentParser


# Parse arguments: 
# * data generator, 
# * ds size
# * epochs
# * lr
# * batch size
# * network
# * width
# * depth
# * activation
# * input shape
# * accuracy
# * loss fn
parser = ArgumentParser()
parser.add_argument("--data", type=str, default="sine")
parser.add_argument("--size", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch", type=int, default=100)
parser.add_argument("--architecture", type=str, default="dense")
parser.add_argument("--width", type=int, default=100)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--ntk_batch", type=int, default=100)
parser.add_argument("--input_shape", type=int, nargs="+")
parser.add_argument("--accuracy", type=bool, default=False)
parser.add_argument("--loss_fn", type=str, default="mean_power(order=2)")
args = parser.parse_args()


ds_switch = {
    "fuel": nl.data.MPGDataGenerator,
    "linear": nl.data.DecisionBoundaryGenerator,
}

generator = ds_switch[args.data](args.size, one_hot=True)
size = args.size
epochs = args.epochs
lr = args.lr
batch_size = args.batch
architecture = args.architecture
width = args.width
depth = args.depth
activation = args.activation
input_shape = tuple(args.input_shape)
accuracy = args.accuracy
loss_fn = args.loss_fn
ntk_batch = args.ntk_batch

# Create model
network_switch = {
    "dense": DenseNetwork,
    "conv": ConvolutionalNetwork,
    "perceptron": Perceptron
}

activation_switch = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid
}

loss_fn_switch = {
    "mean_power": nl.loss_functions.MeanPowerLoss(order=2),
    "cross_entropy": nl.loss_functions.CrossEntropyLoss()
}

network = network_switch[architecture](
    width=width,
    # depth=depth,
    activation=activation_switch[activation]
)
model = nl.models.FlaxModel(
    flax_module=network,
    optimizer=optax.sgd(learning_rate=lr),
    input_shape=input_shape,
    batch_size=ntk_batch
)

# Prepare the recorders
rnd_name = onp.random.randint(0, 100000)
name_prefix = f"{architecture}_{width}_{depth}_{activation}_{size}_{epochs}_{lr}_{batch_size}_{rnd_name}_{loss_fn.__class__.__name__}"
train_recorder = nl.training_recording.JaxRecorder(
    name=f"train_{name_prefix}",
    loss=True,
    entropy=True,
    trace=True,
    accuracy=accuracy,
    magnitude_variance=True,
    update_rate=1,
)
test_recorder = nl.training_recording.JaxRecorder(
    name=f"test_{name_prefix}", 
    loss=True,
    accuracy=accuracy, 
    update_rate=1,
)
train_recorder.instantiate_recorder(data_set=generator.train_ds)
test_recorder.instantiate_recorder(data_set=generator.test_ds)

if accuracy:
    accuracy_fn = nl.accuracy_functions.LabelAccuracy()
else:
    accuracy_fn = None

# Set up the trainer
trainer = nl.training_strategies.SimpleTraining(
        model=model,
        loss_fn=nl.loss_functions.CrossEntropyLoss(),
        accuracy_fn=accuracy_fn,
        recorders=[train_recorder, test_recorder],
    )

# Train the model
trainer.train_model(
    epochs=epochs,
    train_ds=generator.train_ds,
    test_ds=generator.test_ds,
    batch_size=batch_size
)
