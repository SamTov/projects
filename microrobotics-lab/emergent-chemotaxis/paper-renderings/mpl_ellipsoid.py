import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')

def compute_ellipsoid_coordinates(aspect_ratio, base_radius):
    """
    Compute coordinates with which to build an ellipsoid.
    """
    # Compute radii of the ellipsoid
    r_eq = base_radius / np.cbrt(aspect_ratio)
    r_ax = aspect_ratio * r_eq

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = r_eq * np.outer(np.cos(u), np.sin(v))
    y = r_eq * np.outer(np.sin(u), np.sin(v))
    z = r_ax * np.outer(np.ones_like(u), np.cos(v))

    return x, y, z

# Plot sphere
sp_x, sp_y, sp_z = compute_ellipsoid_coordinates(1, 1)
ax.plot_surface(sp_x - 3, sp_y, sp_z,  rstride=4, cstride=4, color='b')

# Plot prolate spheroid
pr_x, pr_y, pr_z = compute_ellipsoid_coordinates(3, 1)
ax.plot_surface(pr_x, pr_y, pr_z,  rstride=4, cstride=4, color='g')

# Plot oblate spheroid
ob_x, ob_y, ob_z = compute_ellipsoid_coordinates(1 / 3, 1)
ax.plot_surface(ob_x + 6, ob_y, ob_z,  rstride=4, cstride=4, color='r')

# Adjust the limits, ticks and view angle

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_zlim(-7, 7)
ax.view_init(36, 26)


plt.show()
