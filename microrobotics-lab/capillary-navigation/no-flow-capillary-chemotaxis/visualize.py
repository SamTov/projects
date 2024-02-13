import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import znvis as vis
import h5py as hf
import sys

prefix = sys.argv[1]
with hf.File(f"{prefix}/trajectory.hdf5") as db:
    agents = db["colloids"]["Unwrapped_Positions"][:, :50, :]
    blood_cells = db["colloids"]["Unwrapped_Positions"][:, 50:, :]

Lx, Ly, Lz = 200, 184.12698412698413, 200

blood_cells[:, :, 0] %= Lx
blood_cells[:, :, 1] %= Ly
blood_cells[:, :, 2] %= Lz

agents[:, :, 0] %= Lx
agents[:, :, 1] %= Ly
agents[:, :, 2] %= Lz

colloid_material = vis.Material(colour=np.array([255, 215, 0]) / 255)
colloid_mesh = vis.Sphere(radius=1.0, material=colloid_material, resolution=4)
colloid_particle = vis.Particle(name="Colloid", mesh=colloid_mesh, position=agents)

blood_material = vis.Material(colour=np.array([136, 8, 8]) / 255)
blood_mesh = vis.Sphere(radius=1.5, material=blood_material, resolution=4)
blood_particle = vis.Particle(name="Blood", mesh=blood_mesh, position=blood_cells)

# Run the visualizer
visualizer = vis.Visualizer(particles=[colloid_particle, blood_particle], frame_rate=80)
visualizer.run_visualization()
