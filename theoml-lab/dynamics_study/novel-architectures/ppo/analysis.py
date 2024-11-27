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

import ml_collections
import matplotlib.pyplot as plt
from compute_ntk import get_ntk_fn, full_ntk_matrix


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
    model = models.Actor(num_outputs=num_actions)

    # Load the data
    data = np.load("data.npy", allow_pickle=True)

    # ntk_fn = nt.batch(
    #     nt.empirical_ntk_fn(model.apply),
    #     2
    # )

    ntk_fn = get_ntk_fn(model.apply)

    root = "/work/stovey/novely-model-study/ppo/model_"
    restore_states = [
        f"{root}39061",
        f"{root}0",
        f"{root}19500"
    ]

    data_point = data[0:4].astype(np.float32)
    print(data_point)
    state = load_state(orbax_checkpointer, restore_states[1])
    actor_ntk = full_ntk_matrix(state['params'], data_point, ntk_fn, 2)

    # actor_ntk, critic_ntk = ntk_fn(data_point, data_point, {'params': state['params']})

    ae, at = compute_cvs(actor_ntk)
    # ce, ct = compute_cvs(critic_ntk)

    print(f"Actor: {ae}, {at}")
    # print(f"Critic: {ce}, {ct}")
  