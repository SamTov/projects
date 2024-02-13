import numpy as np
import znvis as vis
import h5py as hf
import sys

with hf.File(f"test_sim/trajectory.hdf5") as db:
    agents = db["colloids"]["Unwrapped_Positions"][:]

colloid_material = vis.Material(colour=np.array([30, 144, 255]) / 255)
colloid_mesh = vis.Sphere(radius=2.14, material=colloid_material, resolution=4)
colloid_particle = vis.Particle(name="Colloid", mesh=colloid_mesh, position=agents)

# Run the visualizer
visualizer = vis.Visualizer(particles=[colloid_particle], frame_rate=80)
visualizer.run_visualization()
