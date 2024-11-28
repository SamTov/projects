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

import glob
import numpy as np

from rich.progress import track


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.001
  config.latents = 20
  config.batch_size = 784
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
        )[1]

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

    entropies = []
    traces = []
    loss = []

    files = np.sort(glob.glob("/work/stovey/novely-model-study/vae/model_*"))
    nums = [int(item.split("/")[-1].split("_")[-1]) for item in files]
    indices = np.argsort(nums)
    n_sub_samples = 100
    ds =np.array([item for item in next(train_dataset)])
    
    for item in track(files[indices]):
        sub_entropies = []
        sub_traces = []
        state = load_state(orbax_checkpointer, item)

        rng, z_key, eval_rng = random.split(rng, 3)
        z = random.normal(z_key, (64, config.latents))
        # vae = models.model(config.latents)

        # recon_images, mean, logvar = vae.apply(
        #     {"params": state["params"]}, ds, rng
        # )

        loss.append(eval_f(
           state["params"], ds, z, eval_rng, config.latents
        )[0]["loss"] / 784
        )    

        # for _ in range(n_sub_samples):
            
        #     ds_indices = np.random.choice(np.shape(ds)[0], 10, replace=False)
        #     test_ds = jnp.take(
        #         jnp.array(ds), ds_indices, axis=0
        #     )
        #     ntk = ntk_fn(
        #         test_ds,
        #         test_ds,
        #         state["params"]
        #     )


        #     e, t = compute_cvs(ntk)
        #     sub_entropies.append(e)
        #     sub_traces.append(t)

        # entropies.append(
        #     [np.mean(sub_entropies), np.std(sub_entropies)]
        # )
        # traces.append(
        #     [np.mean(sub_traces), np.std(sub_traces)]
        # )

    # np.save("entropy.npy", entropies)
    # np.save("traces.npy", traces)

    # print(entropies)
    # print(traces)

    np.save("losses.npy", loss)
        