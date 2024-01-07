import os
import numpy as np
import znvis as vis
import h5py as hf
import sys

prefix=sys.argv[1]

with hf.File(f"{prefix}/trajectory.hdf5") as db:
    agents = db["colloids"]["Unwrapped_Positions"][:, :50, :][:] - np.array([500., 500., 0.])
    rod = db["colloids"]["Unwrapped_Positions"][:, 50:, :][:] - np.array([500., 500., 0.])

colloid_material = vis.Material(colour=np.array([30, 144, 255]) / 255)
colloid_mesh = vis.Sphere(radius=2.14, material=colloid_material, resolution=5)
colloid_particle = vis.Particle(name="Colloid", mesh=colloid_mesh, position=agents)

# Create rod colloid mesh
rod_material = vis.Material(colour=np.array([255, 140, 0]) / 255)
rod_mesh = vis.Sphere(radius=2.14, material=rod_material, resolution=5)
rod_particle = vis.Particle(name="Rod", mesh=rod_mesh, position=rod)

# Run the visualizer
visualizer = vis.Visualizer(particles=[rod_particle, colloid_particle], frame_rate=80)
visualizer.run_visualization()
