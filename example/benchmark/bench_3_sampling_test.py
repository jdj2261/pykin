import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.scene.scene_manager import SceneManager
from pykin.action.place import PlaceAction
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as p_utils

table_mesh = get_object_mesh('ben_table.stl')
table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]

clearbox_8_mesh = get_object_mesh('clearbox_8.stl')
clearbox_16_mesh = get_object_mesh('clearbox_16.stl')
clearbox_pose = Transform(pos=np.array([0.6, 0.25, table_height + 0.0607473]))

param = {'stack_num' : 6}
benchmark_config = {3 : param}

scene_mngr = SceneManager("visual", is_pyplot=True, benchmark=benchmark_config)
for i in range(20):
    clearbox_1_name = 'clearbox_' + str(i)
    clearbox_1_mesh = get_object_mesh(f'clearbox_{i}' + '.stl', scale=0.9)
    scene_mngr.add_object(name=clearbox_1_name, gtype="mesh", h_mat=clearbox_pose.h_mat, gparam=clearbox_1_mesh, color=[0.8 + i*0.01, 0.8 + i*0.01, 0.8 + i*0.01])

place_action = PlaceAction(scene_mngr, n_samples_held_obj=0, n_samples_support_obj=100)
surface_points_for_support_obj = list(place_action.get_surface_points_for_support_obj("clearbox_8"))
fig, ax = p_utils.init_3d_figure(figsize=(10,6), dpi=120, name="Sampling Object")
for point, normal, (min_x, max_x, min_y, max_y) in surface_points_for_support_obj:
    if not (min_x <= point[0] <= max_x):
        continue
    if not (min_y <= point[1] <= max_y):
        continue
    place_action.scene_mngr.render.render_point(ax, point)

surface_points_for_support_obj = list(place_action.get_surface_points_for_support_obj("clearbox_16"))
for point, normal, (min_x, max_x, min_y, max_y) in surface_points_for_support_obj:
    if not (min_x <= point[0] <= max_x):
        continue
    if not (min_y <= point[1] <= max_y):
        continue
    place_action.scene_mngr.render.render_point(ax, point)

p_utils.plot_basis(ax)
place_action.scene_mngr.render_objects(ax, alpha=0.5)
place_action.show()