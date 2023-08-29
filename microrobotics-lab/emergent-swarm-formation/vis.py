import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import znvis as vis
import h5py as hf
prefix="/data/stovey/embedding_representation_data/predator-prey"

with hf.File(f"{prefix}/5_dimensions/4/deployment/trajectory.hdf5") as db:
	agents = db["colloids"]["Unwrapped_Positions"][:, :100, :]
	rod = db["colloids"]["Unwrapped_Positions"][:, 100:, :]

colloid_material = vis.Material(colour=np.array([30, 144, 255]) / 255)
colloid_mesh = vis.Sphere(radius=4.0, material=colloid_material, resolution=4)
colloid_particle = vis.Particle(name="Colloid", mesh=colloid_mesh, position=agents)

# Create rod colloid mesh
rod_material = vis.Material(colour=np.array([255, 140, 0]) / 255)
rod_mesh = vis.Sphere(radius=2.00, material=rod_material, resolution=4)
rod_particle = vis.Particle(name="Rod", mesh=rod_mesh, position=rod)

# Run the visualizer
visualizer = vis.Visualizer(particles=[colloid_particle, rod_particle], frame_rate=80)
visualizer.run_visualization()
