import functools
from typing import Any, Callable

from absl import logging
import flax
from flax import linen as nn
import agent
import models
import test_episodes
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint

import env_utils
import models
import ppo_lib


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
    config.num_agents = 1
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


def get_experience(
    state: train_state.TrainState,
    simulators: list[agent.RemoteSimulator],
    steps_per_actor: int,
):
  """Collect experience from agents.

  Runs `steps_per_actor` time steps of the game for each of the `simulators`.
  """
  all_experience = []
  # Range up to steps_per_actor + 1 to get one more value needed for GAE.
  for _ in range(steps_per_actor + 1):
    sim_states = []
    for sim in simulators:
      sim_state = sim.conn.recv()
      sim_states.append(sim_state)
    sim_states = np.concatenate(sim_states, axis=0)
    log_probs, values = agent.policy_action(
        state.apply_fn, state.params, sim_states
    )
    log_probs, values = jax.device_get((log_probs, values))
    probs = np.exp(np.array(log_probs))
    for i, sim in enumerate(simulators):
      probabilities = probs[i]
      action = np.random.choice(probs.shape[1], p=probabilities)
      sim.conn.send(action)
    experiences = []
    rewards = []
    for i, sim in enumerate(simulators):
      sim_state, action, reward, done = sim.conn.recv()
      value = values[i, 0]
      log_prob = log_probs[i][action]
      sample = agent.ExpTuple(sim_state, action, reward, value, log_prob, done)
      experiences.append(sample)
      rewards.append(reward)
    all_experience.append(sim_states)
  return all_experience


@functools.partial(jax.jit, static_argnums=1)
def get_initial_params(key: jax.Array, model: nn.Module):
    input_dims = (1, 84, 84, 4)  # (minibatch, height, width, stacked frames)
    init_shape = jnp.ones(input_dims, jnp.float32)
    initial_params = model.init(key, init_shape)['params']
    return initial_params

def create_train_state(
    params,
    model: nn.Module,
    config: ml_collections.ConfigDict,
    train_steps: int,
) -> train_state.TrainState:
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=train_steps,
        )
    else:
        lr = config.learning_rate
    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return state


if __name__ == "__main__":
    # Get the configuration
    config = get_config()

    # Instantiate the checkpointer
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()

    # Load the game
    game = config.game + 'NoFrameskip-v4'
    num_actions = env_utils.get_num_actions(game)

    # Load the model
    model = models.ActorCritic(num_outputs=num_actions)

    # Create an agent
    simulators = [agent.RemoteSimulator(game) for _ in range(config.num_agents)]
    
    loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
    iterations_per_step = (
        config.num_agents * config.actor_steps // config.batch_size
    )

    # We restore from 3 different checkpoints to generate a better state description.
    root = "/work/stovey/novely-model-study/ppo/model_"
    restore_states = [
        f"{root}39061",
        f"{root}0",
        f"{root}19500"
    ]
    sim_states = []
    for experience in restore_states:
        params = orbax_checkpointer.restore(experience)
        state = create_train_state(
            params["model"]["params"],
            model,
            config,
            loop_steps * config.num_epochs * iterations_per_step,
        )

        sim_states.append(get_experience(state, simulators, 500))

    np.save("data.npy", np.concatenate(sim_states, axis=0).squeeze())