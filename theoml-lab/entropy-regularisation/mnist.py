# ZnNL Imports
import znnl as nl

# NN Imports
import flax.linen as nn

# Linalg help
import numpy as onp

# Module imports
from modules import *
from observables import *

# Plotting
import matplotlib.pyplot as plt

# Tracking helpers
from rich.progress import track


class MLP(nn.Module):
    """
    Classic dense NN.
    """

    @nn.compact
    def __call__(self, x):
        """
        Forward pass.
        """
        # Reshape the image
        x = x.reshape((x.shape[0], -1))

        # Dense layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)

        return x
    

generator = nl.data.MNISTGenerator(100)

epochs = 50

init_rng = jax.random.PRNGKey(onp.random.randint(76325426354))
state = create_train_state(MLP(), init_rng, 1e-2)

train_losses = []
test_losses = []

train_entropy = []
test_entropy = []

ntk_fn = get_ntk_function(MLP().apply)
for epoch in track(range(epochs)):
    state, loss = train_epoch(state, generator, 64)
    train_losses.append(loss)
    test_losses.append(
        compute_loss(state, generator.test_ds)
    )
    train_ntk = ntk_fn(
        generator.train_ds["inputs"],
        generator.train_ds["inputs"],
        {"params": state.params}
    )
    train_entropy.append(
        compute_entropy(train_ntk)
    )

    test_ntk = ntk_fn(
        generator.test_ds["inputs"],
        generator.test_ds["inputs"],
        {"params": state.params}
    )
    test_entropy.append(
        compute_entropy(test_ntk)
    )



plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.yscale("log")
plt.ylabel("Train Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.plot(train_entropy, label="Train")
plt.plot(test_entropy, label="Test")
plt.ylabel("Entropy")
plt.xlabel("Epoch")
plt.legend()
plt.show()