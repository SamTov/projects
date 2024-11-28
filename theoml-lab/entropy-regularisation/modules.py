# NN libraries
import jax
import jax.numpy as np
import flax.linen as nn
from flax.training import train_state
import optax
import neural_tangents as nt

# Management modules
from rich.progress import track

# Linalg libraries
import numpy as onp


def create_train_state(module, rng, learning_rate):
    """Creates an initial TrainState."""

    params = module.init(rng, np.ones([1, 28, 28, 1]))['params']

    tx = optax.adam(learning_rate)

    return train_state.TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['inputs'])

        loss = optax.softmax_cross_entropy(logits, batch["targets"]).mean()

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def entropy_reg_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['inputs'])

        loss = optax.softmax_cross_entropy(logits, batch["targets"]).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_epoch(state, generator, batch_size: int):
    """
    Train and epoch.
    """
    train_losses = []

    # Perform training steps over batches
    train_ds_size = len(generator.train_ds["inputs"])
    steps_per_epoch = train_ds_size // batch_size

    # Prepare the shuffle.
    rng = jax.random.PRNGKey(onp.random.randint(764357264952))
    permutations = jax.random.permutation(rng, train_ds_size)
    permutations = np.array_split(permutations, steps_per_epoch)

    # Step over items in batch.
    for permutation in permutations:
        batch = {k: v[permutation, ...] for k, v in generator.train_ds.items()}
        state, loss = train_step(state, batch)
        train_losses.append(loss)

    return state, onp.mean(train_losses)

def regularisation_epoch(state, generator, batch_size: int):
    """
    Train and epoch.
    """
    train_losses = []

    # Perform training steps over batches
    train_ds_size = len(generator.train_ds["inputs"])
    steps_per_epoch = train_ds_size // batch_size

    # Prepare the shuffle.
    rng = jax.random.PRNGKey(onp.random.randint(764357264952))
    permutations = jax.random.permutation(rng, train_ds_size)
    permutations = np.array_split(permutations, steps_per_epoch)

    # Step over items in batch.
    for permutation in permutations:
        batch = {k: v[permutation, ...] for k, v in generator.train_ds.items()}
        state, loss = entropy_reg_step(state, batch)
        train_losses.append(loss)

    return state, onp.mean(train_losses)

