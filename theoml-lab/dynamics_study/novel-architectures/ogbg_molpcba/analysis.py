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

# Helper functions
from rich import print


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


def get_ntk_apply_fn(network, ds):
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
            inputs = np.take(ds, inputs)
            return network.apply(
                params, jraph.batch(inputs), rngs=rng
            ).globals

        return apply_fn

if __name__ == "__main__":
    # For repeated use
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    base_path = "/work/stovey/novely-model-study/ogbg_molpcba/"

    config = get_config()  # Load the config

    # Create the model
    network = create_model(config, deterministic=False)

    # Load the data
    train_graphs = jraph.unbatch(load_datasets(config))[0:20]

    state = load_state(orbax_checkpointer, base_path + "/model_1950")
    
    apply_fn = get_ntk_apply_fn(network, train_graphs)
    
    ntk_fn = nt.batch(
        nt.empirical_ntk_fn(apply_fn),
        2
    )

    indices = jnp.array([i for i in range(len(train_graphs))])

    ntk = ntk_fn(
        indices,
        indices,
        state["params"]
    )

    # eigs, _ = jnp.linalg.eigh(ntk)
    # eigs /= eigs.sum()
    # eigs = jnp.clip(eigs, 1e-12, None)

    # print(f"Entropy: {-1 * (eigs * jnp.log(eigs)).sum()}")
    # print(f"Trace: {jnp.trace(ntk)}")

    # print(network.apply(state["params"], train_graphs, rngs=rng).globals)

    
    


