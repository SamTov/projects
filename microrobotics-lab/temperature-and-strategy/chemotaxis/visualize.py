import znvis as vis
import numpy as np
import h5py as hf


temperature = 150
ensemble = 1
file_path = f"{temperature}K/{ensemble}/deployment/trajectory.hdf5"


def main():
    """
    Run the visuaizer.
    """

    with hf.File(file_path, "r") as db:
        data = db["colloids"]["Unwrapped_Positions"][::10]

    mesh = vis.Sphere(radius=2.0, colour=np.array([30, 144, 255]) / 255, resolution=20)
    particle = vis.Particle(name="colloid", mesh=mesh, position=data)

    visualizer = vis.Visualizer(particles=[particle], frame_rate=80)
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
