import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import neural_tangents as nt
import orbax.checkpoint

import ml_collections
import matplotlib.pyplot as plt

from train import *


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
  ntk_fn = nt.batch(
        nt.empirical_ntk_fn(model.apply),
        2
    )
  
  state = load_state(orbax_checkpointer, "/work/stovey/novely-model-study/vae/model_14")
  test_ds = next(train_dataset)[0:10]
  ntk_fn(test_ds, test_ds, state["params"])

  