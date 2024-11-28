import mdsuite as mds


md_project = mds.Project(
    name="ballistic-diamond", storage_path='/work/stovey'
)

experiment = md_project.add_experiment(
    name="simulation", 
    timestep=0.0001, 
    temperature=300.0, 
    units="metal", 
    simulation_data="/work/stovey/traj.lammpstraj"
)
