import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def plot_cylinder(ax=None, length=1.0, radius=1.0, thickness=0.0,
                  A2B=np.eye(4), ax_s=1, wireframe=True, n_steps=100,
                  alpha=1.0, color="k"):
    """Plot cylinder.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    length : float, optional (default: 1)
        Length of the cylinder

    radius : float, optional (default: 1)
        Radius of the cylinder

    thickness : float, optional (default: 0)
        Thickness of a cylindrical shell. It will be subtracted from the
        outer radius to obtain the inner radius. The difference must be
        greater than 0.

    A2B : array-like, shape (4, 4)
        Center of the cylinder

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of cylinder and surface otherwise

    n_steps : int, optional (default: 100)
        Number of discrete steps plotted in each dimension

    alpha : float, optional (default: 1)
        Alpha value of the cylinder that will be plotted

    color : str, optional (default: black)
        Color in which the cylinder should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis

    Raises
    ------
    ValueError
        If thickness is <= 0
    """


    ax = plt.subplot(111, projection="3d")

    axis_start = A2B.dot(np.array([0, 0, -0.5 * length, 1]))[:3]
    axis_end = A2B.dot(np.array([0, 0, 0.5 * length, 1]))[:3]
    print(axis_start, axis_end)
    axis = axis_end - axis_start
    axis /= length
    print(axis)
    not_axis = np.array([1, 0, 0])
    if (axis == not_axis).all():
        not_axis = np.array([0, 1, 0])

    n1 = np.cross(axis, not_axis)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis, n1)

    t = np.array([0, length])
    # print(t)
    theta = np.linspace(0, 2 * np.pi, n_steps)
    print(theta)
    t, theta = np.meshgrid(t, theta)
    

    X, Y, Z = [axis_start[i] + axis[i] * t
                + radius * np.sin(theta) * n1[i]
                + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

    return ax


ax = plot_cylinder(length=1.0, radius=0.2, thickness=0,
                   wireframe=False, alpha=0.2)

ax.set_xlim((0, 1.5))
ax.set_ylim((-1.5, 0))
ax.set_zlim((-0.8, 0.7))
plt.show()
