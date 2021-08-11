import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from pprint import pprint
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)

filename = '../../asset/urdf/baxter/meshes/base/pedestal_link_collision.stl'

def plot_mesh(ax=None, filename=None, A2B=np.eye(4),
              s=np.array([1.0, 1.0, 1.0]), ax_s=1, wireframe=False,
              convex_hull=False, alpha=1.0, color="k"):

    ax = plt.subplot(111, projection="3d")
    mesh = trimesh.load(filename)
    if convex_hull:
        mesh = mesh.convex_hull
    vertices = mesh.vertices * s
    vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    vertices = np.dot(vertices, A2B.T)[:, :3]
    vectors = np.array([vertices[[i, j, k]] for i, j, k in mesh.faces])

    surface = Poly3DCollection(vectors)
    surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)
    return ax


ax = plot_mesh(filename=filename,
               s=5 * np.ones(3), alpha=0.3)

plt.show()
