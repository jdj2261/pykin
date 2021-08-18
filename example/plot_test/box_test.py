import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from pprint import pprint
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
#sys.path.append(pykin_path)


def plot_box(ax=None, size=np.ones(3), A2B=np.eye(4), ax_s=1,
             color="k", alpha=1.0):
    ax = plt.subplot(111, projection="3d")
    corners = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    corners = (corners - 0.5) * size
    PA = np.hstack(
        (corners, np.ones((len(corners), 1))))
    corners = np.dot(PA, A2B.T)[:, :3]

    p3c = Poly3DCollection(np.array([
        [corners[0], corners[1], corners[2]],
        [corners[1], corners[2], corners[3]],

        [corners[4], corners[5], corners[6]],
        [corners[5], corners[6], corners[7]],

        [corners[0], corners[1], corners[4]],
        [corners[1], corners[4], corners[5]],

        [corners[2], corners[6], corners[7]],
        [corners[2], corners[3], corners[7]],

        [corners[0], corners[4], corners[6]],
        [corners[0], corners[2], corners[6]],

        [corners[1], corners[5], corners[7]],
        [corners[1], corners[3], corners[7]],
    ]))
    p3c.set_alpha(alpha)
    p3c.set_facecolor(color)
    ax.add_collection3d(p3c)

    return ax


plot_box( size=[1, 1, 1],  alpha=0.3)

plt.show()
