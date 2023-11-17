import numpy as np
import znvis as vis
import mitsuba as mi

virus_trajectory = np.zeros((10, 1, 3)) - np.array([5., 0., 0.])
virus_orientation = np.zeros((10, 1, 3)) + np.array([0., 0., 1.])

material = vis.Material(colour=np.array([30, 144, 255]) / 255, mitsuba_bsdf=mi.load_dict({'type': 'diffuse',
'reflectance': {
    'type': 'rgb',
    'value': [0.2, 0.25, 0.7]
}}))

virus_mesh = vis.CustomMesh(file="mesh.stl", material=material)
virus = vis.Particle(
    name="Virus",
    mesh=virus_mesh,
    position=virus_trajectory,
    director=virus_orientation,
)

# Construct the visualizer and run
visualizer = vis.Visualizer(particles=[virus], frame_rate=10)
visualizer.run_visualization()
