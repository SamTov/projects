import pathlib
import numpy as np
import pint
import pickle
import espressomd
import shutil
import h5py
import tqdm

from swarmrl.agents import dummy_models
from swarmrl.engine import espresso
from swarmrl.force_functions import ForceFunction
from swarmrl.utils import utils

FLOW_FILE = pathlib.Path("./flowfield_vein_lowRe_image_name_vein_binary_dilated.png_seed_0.pick")
FLOW_PARAMS_FILE = pathlib.Path("./params_vein_lowRe_image_name_vein_binary_dilated.png_seed_0.pick")
OUT_DIR = pathlib.Path("./test_sim")

def assert_no_langevin_friction(system):
    """
    check that we are using a hacked espresso in which the thermostat 
    only produces noise, but no friction: 
    The friction will be added from the flowfield later
    """
    system.thermostat.set_langevin(kT=0.0, gamma=100, gamma_rotation=100, seed=42)

    # make sure we do not already have stuff in the system
    assert len(system.part.all()) == 0
    assert system.time == 0
    ext_force = np.array([1.1, 2.2, 3.3])
    ext_torque = np.array([3.3, 0.0, 0.0])
    rinertia = 4.321
    test_partcl = system.part.add(
        pos=[0, 0, 0],
        mass=1.234,
        rinertia=3 * [rinertia],
        ext_force=ext_force,
        ext_torque=ext_torque,
        rotation=3 * [True],
    )

    t_step_before = np.copy(system.time_step)
    t_step_test = 0.01
    n_steps = 100
    system.time_step = t_step_test
    system.cell_system.skin = 1
    system.integrator.run(n_steps)
    # check acceleration without friction even though thermostat is active
    np.testing.assert_allclose(
        test_partcl.v, ext_force / test_partcl.mass * n_steps * t_step_test, rtol=1e-10
    )
    np.testing.assert_allclose(
        test_partcl.omega_lab, ext_torque / rinertia * n_steps * t_step_test, rtol=1e-10
    )

    # check that random noise is active
    test_partcl.v = 3 * [0.0]
    test_partcl.omega_lab = 3 * [0.0]
    test_partcl.ext_force = 3 * [0.0]
    test_partcl.ext_torque = 3 * [0.0]
    system.thermostat.set_langevin(kT=1.0, gamma=100, gamma_rotation=100, seed=42)
    system.integrator.run(100)
    np.testing.assert_array_less(1e-10, np.linalg.norm(test_partcl.v))
    np.testing.assert_array_less(1e-10, np.linalg.norm(test_partcl.omega_lab))

    test_partcl.remove()
    system.time = 0.0
    system.time_step = t_step_before
    system.thermostat.turn_off()

def add_colloids(runner: espresso.EspressoMD, n_colloids:int, partcl_params:dict, boundary_2d, agrid_x, agrid_y, ureg, n_tries = 10000):
    
    rng = np.random.default_rng(runner.seed)
    for i in range(n_colloids):
        for _ in range(n_tries):
            pos = rng.random(2) * np.copy(runner.system.box_l[:2])
            idx = (pos / np.array([agrid_x.m_as("sim_length"), agrid_y.m_as("sim_length")])).astype(int)
            if not boundary_2d[tuple(idx)]:
                init_angle = 2 * np.pi * rng.random()
                init_direction = [np.cos(init_angle),
                                  np.sin(init_angle),
                                  0.]
                runner.add_colloid_on_point(init_position=ureg.Quantity([pos[0], pos[1], 0], "sim_length"),
                                            init_direction=init_direction,
                                            **partcl_params)
                break
        else:
            raise RuntimeError(f"could not find position for colloid in {n_tries} tries")

def get_flow_and_boundary(flow_results:dict, mean_flow_vel):
    flow_field = flow_results["velocity"]
    mid_idx = flow_field.shape[2]//2
    flow_2d = flow_field[:,:,mid_idx,:]
    # remove z component if there is any
    flow_2d[:,:,2] = 0.
    
    boundary = flow_results["boundary_mask"]
    boundary_2d= boundary[:,:,mid_idx]
       
    flow_norm = np.linalg.norm(flow_2d, axis=2)
    flow_norm[boundary_2d] = np.nan
    mean_flow_vel_is = np.nanmean(flow_norm)
    
    # scale to desired mean
    flow_2d = flow_2d/mean_flow_vel_is * mean_flow_vel
    
    return flow_2d, boundary_2d

def main():
    ureg = pint.UnitRegistry()
    swim_vel = ureg.Quantity(5, "micrometer/second")
    mean_flow_vel = ureg.Quantity(5, "micrometer/second") 
    # choose box_l along x, infer y length from flowfield
    box_l_x = ureg.Quantity(200, "micrometer")
    n_colloids = 50
    seed = 42
    n_slices = 20000
    
    #DELME
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    
    with FLOW_FILE.open("rb") as f_:
        flow_results = pickle.load(f_)
    with FLOW_PARAMS_FILE.open("rb") as f_:
        flow_params = pickle.load(f_)
    
    system = espressomd.System(box_l = 3*[100], time_step = 1)
    #assert_no_langevin_friction(system)

    box_l_y = flow_params["derived_params"]["box_l"][1]/flow_params["derived_params"]["box_l"][0] * box_l_x
    box_l = utils.convert_array_of_pint_to_pint_of_array([box_l_x, box_l_y, box_l_x], ureg)# box_l_z doesn't matter for 2d sim
    agrid_x = box_l_x / flow_params["derived_params"]["n_cells"][0]
    agrid_y = box_l_y / flow_params["derived_params"]["n_cells"][1]
    
    
    params = espresso.MDParams(
        ureg=ureg,
        fluid_dyn_viscosity=ureg.Quantity(8.9e-3, "pascal * second"),
        WCA_epsilon=ureg.Quantity(311.15, "kelvin") * ureg.boltzmann_constant,
        temperature=ureg.Quantity(311.15, "kelvin"),
        box_length=box_l,
        time_step=ureg.Quantity(0.005, "second"),
        time_slice=ureg.Quantity(0.01, "second"),
        write_interval=ureg.Quantity(0.2, "second"),
        thermostat_type="langevin",
    )
    
    runner = espresso.EspressoMD(
        params, out_folder=OUT_DIR, write_chunk_size=1, n_dims=2, seed=seed, system=system
    )

    flow_2d, boundary_2d = get_flow_and_boundary(flow_results, mean_flow_vel)
    
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

    active_ang_vel = ureg.Quantity(4 * np.pi / 180, "1/second")

    target_momentum_relaxation_timescale = ureg.Quantity(0.01, "second")
    partcl_mass = gamma * target_momentum_relaxation_timescale
    partcl_rinertia = gamma_rot * target_momentum_relaxation_timescale
    
    # add colloids manually, so we know they are in the fluid zone

    partcl_params = {"radius_colloid": partcl_radius,
                     "type_colloid": 0,
                     "gamma_translation": gamma,
                     "gamma_rotation":gamma_rot,
                     "mass": partcl_mass,
                     "rinertia": utils.convert_array_of_pint_to_pint_of_array(
            3 * [partcl_rinertia], ureg
        )}

    add_colloids(runner, n_colloids, partcl_params, boundary_2d, agrid_x, agrid_y, ureg)

    active_force = fluid_force.m_as("sim_force")
    active_torque = (active_ang_vel * gamma_rot).m_as("sim_torque")

    model = dummy_models.ConstForceAndTorque(
        active_force, [0, 0, active_torque]
    )

    force_fn = ForceFunction({"0": model})

    
    for _ in tqdm.tqdm(range(100)):
        runner.integrate(n_slices//100, force_model=force_fn)
    
    params_to_save = {"box_l": box_l.m_as("micrometer"),}
    with (runner.out_folder / "params.pick").open("wb") as f_:
        pickle.dump(params_to_save, f_)
    
    print(f"done. wrote to {runner.out_folder}")
    
    plot_trajectories()
    
def plot_trajectories():
    import matplotlib.pyplot as plt
    
    with FLOW_FILE.open("rb") as f_:
        flow_results = pickle.load(f_)
    with h5py.File(OUT_DIR / "trajectory.hdf5") as traj_file:
        positions = np.array(traj_file["colloids/Unwrapped_Positions"][:, :, :2])
    with (OUT_DIR  / "params.pick").open("rb") as f_:
        params = pickle.load(f_)
    
    flow_2d, boundary_2d = get_flow_and_boundary(flow_results, 5)
    
    flow_norm = np.linalg.norm(flow_2d, axis=2)
    flow_norm[boundary_2d] = np.nan

    box_l = params["box_l"][:2]
    
    positions -= np.floor(positions/box_l[None,None,:]) * box_l[None,None,:]
    
    
    xs, ys = flow_results["xs"], flow_results["ys"]
    xs *= box_l[0]/xs.max()
    ys *= box_l[1]/ys.max()
    
    
    fig, ax = plt.subplots()
    ax.pcolormesh(xs, ys, flow_norm.T)
    ax.plot(positions[0, :, 0], positions[0, :, 1], "o")
    ax.set_aspect("equal")
    fig.savefig(OUT_DIR / "trajectory.png")
    
    plt.show()
    


if __name__ == "__main__":
    main()

