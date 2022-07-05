import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.scene.scene_manager import SceneManager
from pykin.action.place import PlaceAction
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as p_utils

shelf_pose = Transform(pos=np.array([0.9, 0, 1.41725156]),rot=np.array([0, 0, np.pi/2]))
bottle_mesh = get_object_mesh('bottle.stl')
bottle_pose1 = Transform(pos=np.array([1.05, 0, 1.29]))
bottle_pose2 = Transform(pos=np.array([1.0, 0.05, 1.29]))
bottle_pose3 = Transform(pos=np.array([1.0, -0.05,1.29]))
bottle_pose4 = Transform(pos=np.array([0.95, 0.1, 1.29]))
bottle_pose5 = Transform(pos=np.array([0.95, 0, 1.29]))
bottle_pose6 = Transform(pos=np.array([0.95, -0.1, 1.29]))

shelf_9_mesh = get_object_mesh('shelf_9.stl')
shelf_15_mesh = get_object_mesh('shelf_15.stl')

param = {'stack_num' : 6}
benchmark_config = {1 : param}

scene_mngr = SceneManager("visual", is_pyplot=True, benchmark=benchmark_config)
scene_mngr.add_object(name="shelf_9", gtype="mesh", gparam=shelf_9_mesh, h_mat=shelf_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="shelf_15", gtype="mesh", gparam=shelf_15_mesh, h_mat=shelf_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="bottle_1", gtype="mesh", h_mat=bottle_pose1.h_mat, gparam=bottle_mesh, color=[1, 0, 0])
scene_mngr.add_object(name="bottle_2", gtype="mesh", h_mat=bottle_pose2.h_mat, gparam=bottle_mesh, color=[0, 1, 0])
scene_mngr.add_object(name="bottle_3", gtype="mesh", h_mat=bottle_pose3.h_mat, gparam=bottle_mesh, color=[0, 1, 0])
scene_mngr.add_object(name="bottle_4", gtype="mesh", h_mat=bottle_pose4.h_mat, gparam=bottle_mesh, color=[0, 1, 0])
scene_mngr.add_object(name="bottle_5", gtype="mesh", h_mat=bottle_pose5.h_mat, gparam=bottle_mesh, color=[0, 1, 0])
scene_mngr.add_object(name="bottle_6", gtype="mesh", h_mat=bottle_pose6.h_mat, gparam=bottle_mesh, color=[0, 1, 0])

place_action = PlaceAction(scene_mngr, n_samples_held_obj=0, n_samples_support_obj=100)
surface_points_for_support_obj = list(place_action.get_surface_points_for_support_obj("shelf_9"))
fig, ax = p_utils.init_3d_figure(figsize=(10,6), dpi=120, name="Sampling Object")
for point, normal, (min_x, max_x, min_y, max_y) in surface_points_for_support_obj:
    if not (min_x <= point[0] <= max_x):
        continue
    if not (min_y <= point[1] <= max_y):
        continue
    place_action.scene_mngr.render.render_point(ax, point)

surface_points_for_support_obj = list(place_action.get_surface_points_for_support_obj("shelf_15"))
for point, normal, (min_x, max_x, min_y, max_y) in surface_points_for_support_obj:
    if not (min_x <= point[0] <= max_x):
        continue
    if not (min_y <= point[1] <= max_y):
        continue
    place_action.scene_mngr.render.render_point(ax, point)
p_utils.plot_basis(ax)
place_action.scene_mngr.render_objects(ax, alpha=0.1)
place_action.show()