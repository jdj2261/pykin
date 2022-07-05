import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)
from pykin.scene.scene_manager import SceneManager
import pykin.utils.plot_utils as p_utils

scene_mngr = SceneManager("visual", is_pyplot=False)
# scene_mngr.add_object(name="point", gtype="sphere", gparam=0.01, color=[1., 0., 0.])

fig, ax = p_utils.init_3d_figure(figsize=(10,6), dpi=120, name="Sampling Object")
p_utils.plot_basis(ax)
scene_mngr.render_objects(ax, alpha=0.5)
scene_mngr.render.render_point(pose=np.eye(4), radius=1.0)
scene_mngr.show()
# trimesh_scene = trimesh.Scene()
# point = trimesh.Trimesh()
# trimesh_scene.add_geometry(mesh)

