"""
Template of the experiment.
"""
import znnl as nl

import sys
sys.path.append('/home/st/st_st/st_ac134186/work/projects/theoml-lab/universal-dynamics/src')

from generator_builder import build_generator
from loss_fn_selector import select_loss_fn
from optimizer_selector import select_optimizer
from activation_function_selection import select_activation

from network_builder import build_conv_network

import flax.linen as nn


# sed parameters
dataset = DATASET
width = WIDTH
depth = DEPTH
kernel = KERNEL
window = WINDOW
pool_op = nn.max_pool
stride = STRIDE
output = OUTPUT
activation = ACTIVATION
optimizer = OPTIMIZER
ds_size = DSSIZE
one_hot = ONEHOT
learning_rate = LR
input_shape = INPUT
accuracy = ACCURACY
batch_size = BATCH
epochs = EPOCHS
loss = LOSS


# Prepare data
generator = build_generator(dataset, ds_size, one_hot)

# Select a loss function
loss_fn = select_loss_fn(loss)

# Select the optimizer
optimizer = select_optimizer(optimizer)

# Select activation function
activation = select_activation(activation)

# Construct the model
network = build_conv_network(
    width, depth, kernel, stride, window, output, activation, pool_op
)

model = nl.models.FlaxModel(
    flax_module=network, optimizer=optimizer(learning_rate), input_shape=input_shape
)

# Prepare the recorders
train_recorder = nl.training_recording.JaxRecorder(
    name=f"train_recorder",
    loss=True,
    entropy=True,
    trace=True,
    accuracy=accuracy,
    magnitude_variance=True,
    update_rate=1,
)
test_recorder = nl.training_recording.JaxRecorder(
    name=f"test_recorder", loss=True, accuracy=accuracy, update_rate=1
)
train_recorder.instantiate_recorder(data_set=generator.train_ds)
test_recorder.instantiate_recorder(data_set=generator.test_ds)

# Create the trainer
if accuracy:
    trainer = nl.training_strategies.SimpleTraining(
        model=model,
        loss_fn=nl.loss_functions.MeanPowerLoss(order=2),
        accuracy_fn=nl.accuracy_functions.LabelAccuracy(),
        recorders=[train_recorder, test_recorder],
    )
else:
    trainer = nl.training_strategies.SimpleTraining(
        model=model, loss_fn=loss_fn, recorders=[train_recorder, test_recorder]
    )

# Train model and dump recorders
_ = trainer.train_model(
    train_ds=generator.train_ds,
    test_ds=generator.test_ds,
    batch_size=batch_size,
    epochs=epochs,
)

train_recorder.dump_records()
test_recorder.dump_records()
