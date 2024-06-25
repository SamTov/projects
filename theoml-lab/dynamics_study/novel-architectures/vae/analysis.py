import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import neural_tangents as nt
import orbax.checkpoint

import ml_collections
import matplotlib.pyplot as plt

from train import *
from compute_ntk import *


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.001
  config.latents = 20
  config.batch_size = 128
  config.num_epochs = 200
  return config

def load_state(checkpointer, file):
   """ Load a model state. """
   state = checkpointer.restore(file)

   return state["model"]

def get_dataset():
  ds_builder = tfds.builder('binarized_mnist')
  ds_builder.download_and_prepare()

  return input_pipeline.build_train_set(config.batch_size, ds_builder)

def get_apply_fn(config):

    def apply_fn(params, inputs):
        rng = jax.random.key(0)
        rng, init_rng = jax.random.split(rng)
        return models.model(config.latents).apply(
            {"params": params}, inputs, rng
        )[0]

    return apply_fn

def compute_cvs(ntk):

    eigs, _ = np.linalg.eigh(ntk)

    eigs = np.clip(eigs, 1e-11, None)

    eigs /= eigs.sum()

    return -np.sum(eigs * np.log(eigs)), np.trace(ntk)


if __name__ == "__main__":
  
    # Get the config
    config = get_config()
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    # Set rng
    rng = random.key(0)
    rng, key = random.split(rng)

    # Load the dataset
    train_dataset = get_dataset()

    model = models.model(config.latents)
    apply_fn = get_apply_fn(config)

    ntk_fn = get_ntk_fn(apply_fn)



    state = load_state(orbax_checkpointer, "/work/stovey/novely-model-study/vae/model_14")
    test_ds = next(train_dataset)[0:4]
    ntk = full_ntk_matrix(state['params'], test_ds, ntk_fn, 2)

    e, t = compute_cvs(ntk)

    print(e)
    print(t)

