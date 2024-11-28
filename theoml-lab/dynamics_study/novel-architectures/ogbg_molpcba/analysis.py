# Network imports
import jax
import jax.numpy as jnp
import jraph
import orbax.checkpoint
import neural_tangents as nt
import numpy as np

# Custom script imports
import models
from train import *
import input_pipeline
import glob

# Helper functions
from rich import print
import matplotlib.pyplot as plt


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Optimizer.
  config.optimizer = 'adam'
  config.learning_rate = 1e-3

  # Training hyperparameters.
  config.batch_size = 256
  config.num_train_steps = 100_000
  config.log_every_steps = 10
  config.eval_every_steps = 1_000
  config.checkpoint_every_steps = 10_000
  config.add_virtual_node = False
  config.add_undirected_edges = True
  config.add_self_loops = True

  # GNN hyperparameters.
  config.model = 'GraphConvNet'
  config.message_passing_steps = 5
  config.latent_size = 256
  config.dropout_rate = 0.1
  config.num_mlp_layers = 2
  config.num_classes = 128
  config.skip_connections = True
  config.layer_norm = True

  return config


def load_state(checkpointer, file):
   """ Load a model state. """
   state = checkpointer.restore(file)

   return state["model"]


def load_datasets(config):
   """ Load up the datasets. """
   datasets = input_pipeline.get_datasets(
        config.batch_size,
        add_virtual_node=config.add_virtual_node,
        add_undirected_edges=config.add_undirected_edges,
        add_self_loops=config.add_self_loops,
   )

   graphs = jax.tree_util.tree_map(np.asarray, next(iter(datasets['train'])))
   
   return graphs


def compute_cvs(ntk):

    eigs, _ = np.linalg.eigh(ntk)

    eigs = np.clip(eigs, 1e-11, None)

    eigs /= eigs.sum()

    return -np.sum(eigs * np.log(eigs)), np.trace(ntk)


def get_ntk_apply_fn(network):
        """
        Return an NTK capable apply function.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        inputs : np.ndarray
                Feature vector on which to apply the model.

        TODO(Konsti): Make the apply function work with the batch_stats.

        Returns
        -------
        Acts on the data with the model architecture and parameter set.
        """

        def apply_fn(params, inputs):
            rng = jax.random.key(0)
            rng, init_rng = jax.random.split(rng)
            return network.apply(
                params, inputs, rngs=rng
            ).globals

        return apply_fn

if __name__ == "__main__":
    # For repeated use
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()

    config = get_config()  # Load the config

    # Create the model
    network = create_model(config, deterministic=False)

    # Load the data
    ds = load_datasets(config)

    entropies = []
    traces = []
    loss = []

    files = np.sort(glob.glob("/work/stovey/novely-model-study/ogbg_molpcba/model_*"))
    nums = [int(item.split("/")[-1].split("_")[-1]) for item in files]
    indices = np.argsort(nums)
    n_sub_samples = 50
    print(len(indices))


    for item in files[indices][::10]:
        sub_entropies = []
        sub_traces = []
        state = load_state(orbax_checkpointer, item)
        # print(state.apply)
        labels = ds.globals

        graphs = replace_globals(ds)
        rng = jax.random.key(0)
        rngs, init_rng = jax.random.split(rng)

        pred_graphs = network.apply(state["params"], graphs, rngs=rngs)

        logits = pred_graphs.globals
        mask = get_valid_mask(labels, graphs)
        sl = binary_cross_entropy_with_mask(logits=logits, labels=labels, mask=mask)
        ssl = EvalMetrics.single_from_model_output(
        loss=sl, logits=logits, labels=labels, mask=mask
        ).compute()
        # Compute the various metrics.
        loss.append(ssl)

        

        # for _ in range(n_sub_samples):
            
        #     ds_indices = np.random.choice(len(ds), 10, replace=False)

        #     test_ds = [ds[i] for i in ds_indices]

        #     # ntk = ntk_fn(
        #     #     jraph.batch(test_ds),
        #     #     jraph.batch(test_ds),
        #     #     state["params"]
        #     # )

        #     # e, t = compute_cvs(ntk)
        #     # sub_entropies.append(e)
        #     # sub_traces.append(t)

        # entropies.append(
        #     [np.mean(sub_entropies), np.std(sub_entropies)]
        # )
        # traces.append(
        #     [np.mean(sub_traces), np.std(sub_traces)]
        # )

    
    # np.save("entropy.npy", entropies)
    # np.save("traces.npy", traces)
    np.save("losses.npy", loss)

