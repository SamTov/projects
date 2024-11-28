from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow as tf

import env_utils
import models
# import ppo_lib

import neural_tangents as nt
import orbax.checkpoint as ocp
import numpy as np
import jax.numpy as jnp

import ml_collections
import matplotlib.pyplot as plt
from compute_ntk import get_ntk_fn, full_ntk_matrix
from test_episodes import policy_test

import glob


def get_config():
    """Get the default configuration.

    The default hyperparameters originate from PPO paper arXiv:1707.06347
    and openAI baselines 2::
    https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py
    """
    config = ml_collections.ConfigDict()
    # The Atari game used.
    config.game = 'Pong'
    # Total number of frames seen during training.
    config.total_frames = 40000000
    # The learning rate for the Adam optimizer.
    config.learning_rate = 2.5e-4
    # Batch size used in training.
    config.batch_size = 256
    # Number of agents playing in parallel.
    config.num_agents = 8
    # Number of steps each agent performs in one policy unroll.
    config.actor_steps = 128
    # Number of training epochs per each unroll of the policy.
    config.num_epochs = 3
    # RL discount parameter.
    config.gamma = 0.99
    # Generalized Advantage Estimation parameter.
    config.lambda_ = 0.95
    # The PPO clipping parameter used to clamp ratios in loss function.
    config.clip_param = 0.1
    # Weight of value function loss in the total loss.
    config.vf_coeff = 0.5
    # Weight of entropy bonus in the total loss.
    config.entropy_coeff = 0.01
    # Linearly decay learning rate and clipping parameter to zero during
    # the training.
    config.decaying_lr_and_clip_param = True

    return config


def load_state(checkpointer, file):
    """ Load a model state. """
    state = checkpointer.restore(file)

    return state["model"]


def compute_cvs(ntk):

    eigs, _ = np.linalg.eigh(ntk)

    eigs = np.clip(eigs, 1e-11, None)

    eigs /= eigs.sum()

    return -np.sum(eigs * np.log(eigs)), np.trace(ntk)



if __name__ == "__main__":
  
    # Get the configuration
    config = get_config()

    # Instantiate the checkpointer
    orbax_checkpointer = ocp.StandardCheckpointer()

    # Load the game
    game = config.game + 'NoFrameskip-v4'
    num_actions = env_utils.get_num_actions(game)

    # Load the model
    model = models.ActorCritic(num_outputs=num_actions)

    # Load the data
    data = np.load("data.npy", allow_pickle=True)

    ntk_fn = nt.empirical_ntk_fn(model.apply)

    files = np.sort(glob.glob("/work/stovey/novely-model-study/ppo/model_*"))
    nums = [int(item.split("/")[-1].split("_")[-1]) for item in files]
    indices = np.argsort(nums)
    n_sub_samples = 100

    actor_entropies = []
    critic_entropies = []
    actor_traces = []
    critic_traces = []
    rewards = []

    n_sub_samples = 100


    for item in files[indices][::100]:
        sub_actor_entropies = []
        sub_actor_traces = []
        sub_critic_entropies = []
        sub_critic_traces = []

        state = load_state(orbax_checkpointer, item)

        for _ in range(n_sub_samples):
            
            ds_indices = np.random.choice(np.shape(data)[0], 10, replace=False)
            test_ds = jnp.take(jnp.array(data), ds_indices, axis=0)

            actor_ntk, critic_ntk = ntk_fn(test_ds, test_ds, {'params': state['params']})

            ae, at = compute_cvs(actor_ntk)
            ce, ct = compute_cvs(critic_ntk)

            sub_actor_entropies.append(ae)
            sub_actor_traces.append(at)
            sub_critic_entropies.append(ce)
            sub_critic_traces.append(ct)

        actor_entropies.append(
            [np.mean(sub_actor_entropies), np.std(sub_actor_entropies)]
        )
        actor_traces.append(
            [np.mean(sub_actor_traces), np.std(sub_actor_traces)]
        )
        critic_entropies.append(
            [np.mean(sub_critic_entropies), np.std(sub_critic_entropies)]
        )
        critic_traces.append(
            [np.mean(sub_critic_traces), np.std(sub_critic_traces)]
        )

        rewards.append(policy_test(1, model.apply, state['params'], game))

    np.save("actor_entropy.npy", actor_entropies)
    np.save("actor_traces.npy", actor_traces)

    np.save("critic_entropy.npy", critic_entropies)
    np.save("critic_traces.npy", critic_traces)

    np.save("rewards.npy", rewards)
