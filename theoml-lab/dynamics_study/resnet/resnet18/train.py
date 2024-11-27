"""
Module for training the networks and storing data.
"""
# Linalg libraries
import numpy as np

# Neural network libraries
import optax
from transformers import ResNetConfig, FlaxResNetForImageClassification

# ZnNL
import znnl as nl
from znnl.models import HuggingFaceFlaxModel


# Parameters
epochs = 200
lr = 1e-3
batch_size = 128
ntk_batch_size = 25

# Define the generator
generator = nl.data.CIFAR10Generator(10000)

# Input data needs to have shape (num_points, channels, height, width)
train_ds={"inputs": np.swapaxes(generator.train_ds["inputs"], 1, 3), "targets": generator.train_ds["targets"]}
test_ds={"inputs": np.swapaxes(generator.test_ds["inputs"], 1, 3), "targets": generator.test_ds["targets"]}

generator.train_ds = train_ds
generator.test_ds = test_ds

# Create the ResNet18 config
resnet18_config = ResNetConfig(
    num_channels = 3,
    embedding_size = 64, 
    hidden_sizes = [64, 128, 256, 512], 
    depths = [2, 2, 2, 2], 
    layer_type = 'bottleneck', 
    hidden_act = 'relu', 
    downsample_in_first_stage = False, 
    id2label = {i: i for i in range(10)}, # Dummy labels to define the output dimension
    return_dict = True,
)

_model = FlaxResNetForImageClassification(
    config=resnet18_config,
    input_shape=(1, 32, 32, 3),
    seed=0,
    _do_init = True,
)

model = HuggingFaceFlaxModel(
    _model, 
    optax.adam(learning_rate=lr),
    store_on_device=False,
    batch_size=ntk_batch_size,
)

loss_fn = nl.loss_functions.CrossEntropyLoss()

# Prepare the recorders
rnd_name = np.random.randint(0, 100000)
name_prefix = f"resnet18_{rnd_name}"
recorders = []
train_recorder = nl.training_recording.JaxRecorder(
    name=f"train_{name_prefix}",
    loss=True,
    entropy=True,
    trace=True,
    accuracy=True,
    update_rate=1,
)
test_recorder = nl.training_recording.JaxRecorder(
    name=f"test_{name_prefix}", 
    loss=True,
    accuracy=True,
    entropy=True,
    trace=True, 
    update_rate=1,
)
train_recorder.instantiate_recorder(data_set=generator.train_ds)
test_recorder.instantiate_recorder(data_set=generator.test_ds)

recorders.append(train_recorder)
recorders.append(test_recorder)

# Create recorders for each class
# for i in range(10):
#     indices = np.where(
#         np.apply_along_axis(
#             lambda x: np.where(x == 1), 1, generator.train_ds["targets"]
#         ).flatten() == i
#     )
#     ds = {'inputs': generator.train_ds["inputs"][indices], 'targets': generator.train_ds["targets"][indices]}
#     recorder = nl.training_recording.JaxRecorder(
#         name=f"{rnd_name}_class_{i}",
#         loss=True,
#         accuracy=True,
#         entropy=True,
#         trace=True,
#         update_rate=1,
#     )
#     recorder.instantiate_recorder(data_set=ds)
#     recorders.append(recorder)

accuracy_fn = nl.accuracy_functions.LabelAccuracy()

# Set up the trainer
trainer = nl.training_strategies.SimpleTraining(
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        recorders=recorders,
    )

# Train the model
trainer.train_model(
    epochs=epochs,
    train_ds=generator.train_ds,
    test_ds=generator.test_ds,
    batch_size=batch_size
)
