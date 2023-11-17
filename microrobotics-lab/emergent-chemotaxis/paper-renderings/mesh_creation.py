from vedo import Ellipsoid, Arrow, Arrow2D, show, write, Plotter, merge, screenshot
import numpy as np

plt = Plotter()

shaft_radius = 0.01
head_radius = 0.03
head_length = 0.1

resolution = 100

def get_radii(aspect_ratio, base_radius):
    r_eq = base_radius / np.cbrt(aspect_ratio)
    r_ax = aspect_ratio * r_eq

    return r_eq, r_ax


r_eq, r_ax = get_radii(1, 1)
circle = Ellipsoid(
    pos=(0, 0, 0),
    axis1=(r_eq, 0, 0),
    axis2=(0, r_eq, 0),
    axis3=(0, 0, r_ax),
    res=resolution,
)
circle_arrow = Arrow(
    start_pt=(0, 0, 0),
    end_pt=(0, 0, 1.5),
    shaft_radius=shaft_radius,
    head_radius=head_radius,
    head_length=head_length,
    res=resolution,
    c="#ED254E"
)

r_eq = 1 / np.cbrt(3)
r_ax = 3 * r_eq
r_eq, r_ax = get_radii(3, 1)
prolate = Ellipsoid(
    pos=(2, 0, 0),
    axis1=(r_eq, 0, 0),
    axis2=(0, r_eq, 0),
    axis3=(0, 0, r_ax),
    res=resolution,
)
prolate_arrow = Arrow(
    start_pt=(2, 0, 0),
    end_pt=(2, 0, 1.5),
    shaft_radius=shaft_radius,
    head_radius=head_radius,
    head_length=head_length,
    res=resolution,
    c="#ED254E"
)

r_eq, r_ax = get_radii(1 / 3, 1)
oblate = Ellipsoid(
    pos=(4, 0, 0),
    axis1=(r_eq, 0, 0),
    axis2=(0, r_eq, 0),
    axis3=(0, 0, r_ax),
    res=resolution,
)
oblate_arrow = Arrow(
    start_pt=(4, 0, 0),
    end_pt=(4, 0, 1.5),
    shaft_radius=shaft_radius,
    head_radius=head_radius,
    head_length=head_length,
    res=resolution,
    c="#ED254E"
)

plt.azimuth(0)
cam1 = {
    'pos': (0.0, -10.0, 5.0),  # Camera position
    'focalPoint': (2.0, 0.0, 0.0),  # Point to look at
    'viewup': (0, 0, 1),  # Up direction along the Z-axis,
}

plt.show(
    oblate,
    oblate_arrow,
    prolate,
    prolate_arrow,
    circle,
    circle_arrow,
    # axes=1, 
    camera=cam1,
)
