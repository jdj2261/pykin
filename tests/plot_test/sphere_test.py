import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)
import numpy as np
import matplotlib.pyplot as plt



def plot_sphere(ax=None, radius=1.0, p=np.zeros(3), ax_s=1, wireframe=True,
                n_steps=20, alpha=1.0, color="k"):
    ax = plt.subplot(111, projection="3d")
    
    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = p[0] + radius * np.sin(phi) * np.cos(theta)
    y = p[1] + radius * np.sin(phi) * np.sin(theta)
    z = p[2] + radius * np.cos(phi)

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax

ax = plot_sphere(
    radius=0.5, wireframe=False, alpha=0.1, color="k", n_steps=20, ax_s=0.5)

plt.show()
