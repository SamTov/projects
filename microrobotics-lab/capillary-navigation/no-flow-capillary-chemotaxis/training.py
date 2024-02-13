import pathlib
import numpy as np
import pint
import pickle
import espressomd
import shutil
import h5py

import swarmrl as srl
from swarmrl.engine import espresso
from swarmrl.utils import utils
from swarmrl.actions import Action

import optax

from rl_parameters import (
    ChemotaxisObservable,
    ChemotaxisTask,
    field_decay,
    ActorCriticNet,
)


def add_colloids(
    runner: espresso.EspressoMD,
    n_colloids: int,
    partcl_params: dict,
    boundary_2d,
    agrid_x,
    agrid_y,
    ureg,
    n_tries=10000,
):
    rng = np.random.default_rng(runner.seed)
    for p_type in partcl_params:
        for i in range(p_type["n_colloids"]):
            for _ in range(n_tries):
                pos = rng.random(2) * np.copy(runner.system.box_l[:2])
                idx = (
                    pos / np.array([agrid_x.m_as("sim_length"), agrid_y.m_as("sim_length")])
                ).astype(int)
                if not boundary_2d[tuple(idx)]:
                    init_angle = 2 * np.pi * rng.random()
                    init_direction = [np.cos(init_angle), np.sin(init_angle), 0.0]
                    runner.add_colloid_on_point(
                        init_position=ureg.Quantity([pos[0], pos[1], 0], "sim_length"),
                        init_direction=init_direction,
                        **p_type["esp_params"],
                    )
                    break
            else:
                raise RuntimeError(
                    f"could not find position for colloid in {n_tries} tries"
                )


def get_flow_and_boundary(flow_results: dict, mean_flow_vel):
    flow_field = flow_results["velocity"]
    mid_idx = flow_field.shape[2] // 2
    flow_2d = flow_field[:, :, mid_idx, :]
    # remove z component if there is any
    flow_2d[:, :, 2] = 0.0

    boundary = flow_results["boundary_mask"]
    boundary_2d = boundary[:, :, mid_idx]

    flow_norm = np.linalg.norm(flow_2d, axis=2)
    flow_norm[boundary_2d] = np.nan
    mean_flow_vel_is = np.nanmean(flow_norm)

    # scale to desired mean
    flow_2d = flow_2d / mean_flow_vel_is * mean_flow_vel

    return flow_2d, boundary_2d


def get_system_runner(system):
    """
    Helper function for episodic training.
    """
    FLOW_FILE = pathlib.Path(
        "./flowfield_vein_lowRe_image_name_vein_binary_dilated.png_seed_0.pick"
    )
    FLOW_PARAMS_FILE = pathlib.Path(
        "./params_vein_lowRe_image_name_vein_binary_dilated.png_seed_0.pick"
    )
    OUT_DIR = pathlib.Path("/data/stovey/capillary/ep_training/")
    ureg = pint.UnitRegistry()
    swim_vel = ureg.Quantity(5, "micrometer/second")
    mean_flow_vel = ureg.Quantity(5, "micrometer/second")
    # choose box_l along x, infer y length from flowfield
    box_l_x = ureg.Quantity(200, "micrometer")
    seed = np.random.randint(4865)
    OUT_DIR = f"{OUT_DIR}_{seed}"

    with FLOW_FILE.open("rb") as f_:
        flow_results = pickle.load(f_)
    with FLOW_PARAMS_FILE.open("rb") as f_:
        flow_params = pickle.load(f_)

    box_l_y = (
        flow_params["derived_params"]["box_l"][1]
        / flow_params["derived_params"]["box_l"][0]
        * box_l_x
    )
    box_l = utils.convert_array_of_pint_to_pint_of_array(
        [box_l_x, box_l_y, box_l_x], ureg
    )  # box_l_z doesn't matter for 2d sim
    agrid_x = box_l_x / flow_params["derived_params"]["n_cells"][0]
    agrid_y = box_l_y / flow_params["derived_params"]["n_cells"][1]

    params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-3, "pascal * second"),
        WCA_epsilon=ureg.Quantity(311.15, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(311.15, "kelvin"),
        box_length=box_l,
        time_step=ureg.Quantity(0.001, "second"),
        time_slice=ureg.Quantity(0.1, "second"),
        write_interval=ureg.Quantity(0.5, "second"),
        thermostat_type="langevin",
    )

    runner = espresso.EspressoMD(
        params,
        out_folder=OUT_DIR,
        write_chunk_size=1,
        n_dims=2,
        seed=seed,
        system=system,
    )

    flow_2d, boundary_2d = get_flow_and_boundary(flow_results, mean_flow_vel)
    
    n_colloids = 50
    partcl_radius = ureg.Quantity(1, "micrometer")
    gamma = 6 * np.pi * params.fluid_dyn_viscosity * partcl_radius
    gamma_rot = 8 * np.pi * params.fluid_dyn_viscosity * partcl_radius**3

    runner.add_flowfield(
        flow_2d[:, :, np.newaxis, :],
        gamma,
        utils.convert_array_of_pint_to_pint_of_array(
            [agrid_x, agrid_y, box_l[2]], ureg
        ),
    )
    
    fluid_force = gamma * (swim_vel+mean_flow_vel)
    potential = 3*(fluid_force* agrid_x + params.WCA_epsilon) * boundary_2d
    runner.add_external_potential(
        potential[:, :, np.newaxis],
        utils.convert_array_of_pint_to_pint_of_array(
            [agrid_x, agrid_y, params.box_length[2]], ureg
        ),
    )

    target_momentum_relaxation_timescale = ureg.Quantity(0.01, "second")
    partcl_mass = gamma * target_momentum_relaxation_timescale
    partcl_rinertia = gamma_rot * target_momentum_relaxation_timescale

    robot_params = {
        "radius_colloid": partcl_radius,
        "type_colloid": 0,
        "gamma_translation": gamma,
        "gamma_rotation": gamma_rot,
        "mass": partcl_mass,
        "rinertia": utils.convert_array_of_pint_to_pint_of_array(
            3 * [partcl_rinertia], ureg
        ),
    }
    robot_p_params = {"n_colloids": n_colloids, "esp_params": robot_params}


    blood_radii = ureg.Quantity(1.5, "micrometer")
    blood_gamma = 6 * np.pi * params.fluid_dyn_viscosity * blood_radii
    blood_gamma_rot = 8 * np.pi * params.fluid_dyn_viscosity * blood_radii**3
    n_blood_cells = 750

    blood_params = {
        "radius_colloid": blood_radii,
        "type_colloid": 1,
        "gamma_translation": blood_gamma,
        "gamma_rotation": blood_gamma_rot,
        "mass": blood_gamma * target_momentum_relaxation_timescale,
        "rinertia": utils.convert_array_of_pint_to_pint_of_array(
            3 * [blood_gamma_rot * target_momentum_relaxation_timescale], ureg
        ),
    }
    blood_p_params = {"n_colloids": n_blood_cells, "esp_params": blood_params}
    
    add_colloids(runner, n_colloids, [robot_p_params, blood_p_params], boundary_2d, agrid_x, agrid_y, ureg)


    # add_colloids(
    #     runner, n_blood_cells, blood_params, boundary_2d, agrid_x, agrid_y, ureg
    # )

    # Add flow to system
    # runner.add_const_force_to_colloids(
    #     ureg.Quantity(
    #         np.array([0., 3e-16, 0.]), "newton"
    #         ),
    #         type=0
    #     )
    
    # runner.add_const_force_to_colloids(
    #     ureg.Quantity(
    #         np.array([0., 3e-17, 0.]), "newton"
    #         ),
    #         type=1
    #     )

    return runner


def get_system_parameters(system):
    """
    Helper function for episodic training.
    """
    FLOW_FILE = pathlib.Path(
        "./flowfield_vein_lowRe_image_name_vein_binary_dilated.png_seed_0.pick"
    )
    FLOW_PARAMS_FILE = pathlib.Path(
        "./params_vein_lowRe_image_name_vein_binary_dilated.png_seed_0.pick"
    )
    OUT_DIR = pathlib.Path(".ep_training/test_sim")
    ureg = pint.UnitRegistry()
    swim_vel = ureg.Quantity(5, "micrometer/second")
    mean_flow_vel = ureg.Quantity(5, "micrometer/second")
    # choose box_l along x, infer y length from flowfield
    box_l_x = ureg.Quantity(200, "micrometer")
    seed = np.random.randint(4865)
    OUT_DIR = f"{OUT_DIR}_{seed}"

    with FLOW_FILE.open("rb") as f_:
        flow_results = pickle.load(f_)
    with FLOW_PARAMS_FILE.open("rb") as f_:
        flow_params = pickle.load(f_)

    box_l_y = (
        flow_params["derived_params"]["box_l"][1]
        / flow_params["derived_params"]["box_l"][0]
        * box_l_x
    )
    box_l = utils.convert_array_of_pint_to_pint_of_array(
        [box_l_x, box_l_y, box_l_x], ureg
    )  # box_l_z doesn't matter for 2d sim
    agrid_x = box_l_x / flow_params["derived_params"]["n_cells"][0]
    agrid_y = box_l_y / flow_params["derived_params"]["n_cells"][1]

    params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-3, "pascal * second"),
        WCA_epsilon=ureg.Quantity(311.15, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(311.15, "kelvin"),
        box_length=box_l,
        time_step=ureg.Quantity(0.005, "second"),
        time_slice=ureg.Quantity(0.1, "second"),
        write_interval=ureg.Quantity(0.2, "second"),
        thermostat_type="langevin",
    )

    runner = espresso.EspressoMD(
        params,
        out_folder=OUT_DIR,
        write_chunk_size=1,
        n_dims=2,
        seed=seed,
        system=system,
    )

    flow_2d, boundary_2d = get_flow_and_boundary(flow_results, mean_flow_vel)

    partcl_radius = ureg.Quantity(1, "micrometer")
    gamma = 6 * np.pi * params.fluid_dyn_viscosity * partcl_radius
    gamma_rot = 8 * np.pi * params.fluid_dyn_viscosity * partcl_radius**3
    fluid_force = gamma * (swim_vel + mean_flow_vel)
    
    active_ang_vel = ureg.Quantity(4 * np.pi / 180, "1/second")

    target_momentum_relaxation_timescale = ureg.Quantity(0.01, "second")
    partcl_mass = gamma * target_momentum_relaxation_timescale
    partcl_rinertia = gamma_rot * target_momentum_relaxation_timescale

    # add colloids manually, so we know they are in the fluid zone

    partcl_params = {
        "radius_colloid": partcl_radius,
        "type_colloid": 0,
        "gamma_translation": gamma,
        "gamma_rotation": gamma_rot,
        "mass": partcl_mass,
        "rinertia": utils.convert_array_of_pint_to_pint_of_array(
            3 * [partcl_rinertia], ureg
        ),
    }

    runner = None
    active_force = fluid_force.m_as("sim_force")
    active_torque = (active_ang_vel * gamma_rot).m_as("sim_torque")

    return box_l_x, box_l_y, active_force, active_torque


def main():
    system = espressomd.System(box_l=3 * [100], time_step=1)

    box_l_x, box_l_y, active_force, active_torque = get_system_parameters(system)

    # RL setup
    observable = ChemotaxisObservable(
        source=np.array([119.0, 175.0, 0]),
        decay_fn=field_decay,
        box_length=np.array([box_l_x.magnitude, box_l_y.magnitude, box_l_x.magnitude]),
        scale_factor=100,
    )
    task = ChemotaxisTask(
        source=np.array([119.0, 175.0, 0]),
        decay_function=field_decay,
        box_length=np.array([box_l_x.magnitude, box_l_y.magnitude, box_l_x.magnitude]),
        reward_scale_factor=10,
    )

    network = srl.networks.FlaxModel(
        flax_model=ActorCriticNet(),
        optimizer=optax.adam(learning_rate=0.001),
        input_shape=(1,),
    )

    translate = Action(force=active_force)
    rotate_clockwise = Action(torque=np.array([0.0, 0.0, active_torque]))
    rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -active_torque]))
    do_nothing = Action()

    actions = {
        "RotateClockwise": rotate_clockwise,
        "Translate": translate,
        "RotateCounterClockwise": rotate_counter_clockwise,
        "DoNothing": do_nothing,
    }
    loss = srl.losses.ProximalPolicyLoss(entropy_coefficient=0.02, epsilon=0.2)
    agent = srl.agents.ActorCriticAgent(
        particle_type=0,
        network=network,
        task=task,
        observable=observable,
        actions=actions,
        loss=loss,
    )
    rl_trainer = srl.trainers.EpisodicTrainer(
        [agent]
    )
    rewards = rl_trainer.perform_rl_training(
        get_engine=get_system_runner,
        n_episodes=5000,
        system=system,
        reset_frequency=300,
        episode_length=10,
    )
    rl_trainer.export_models()

    np.save("rewards.npy", rewards)


if __name__ == "__main__":
    main()
