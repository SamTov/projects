import numpy as np
import h5py as hf

import znvis as vis

# Load data from the database
with hf.File("training/trajectory.hdf5", 'r') as db:
    prey = db["colloids"]["Unwrapped_Positions"][:, :100, :]
    predators = db["colloids"]["Unwrapped_Positions"][:, 100:, :]


# Create free colloid mesh
prey_material = vis.Material(colour=np.array([30, 144, 255]) / 255)
prey_mesh = vis.Sphere(radius=4.0, material=prey_material, resolution=5)
prey_particle = vis.Particle(name="Prey", mesh=prey_mesh, position=prey)

# Create rod colloid mesh
predator_material = vis.Material(colour=np.array([255, 140, 0]) / 255)
predator_mesh = vis.Sphere(radius=2.0, material=predator_material, resolution=5)
predator_particle = vis.Particle(name="Rod", mesh=predator_mesh, position=predators)

# Run the visualizer
visualizer = vis.Visualizer(particles=[prey_particle, predator_particle], frame_rate=80)
visualizer.run_visualization()
