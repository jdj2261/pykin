import numpy as np
import sys, os



from pykin.kinematics.transform import Transform
from pykin.scene.scene_manager import SceneManager
from pykin.action.place import PlaceAction
from pykin.utils.mesh_utils import get_object_mesh, get_mesh_bounds
import pykin.utils.plot_utils as p_utils

table_mesh = get_object_mesh('ben_table.stl')
table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]

disk_mesh = get_object_mesh('hanoi_disk.stl')
disk_mesh_bound = get_mesh_bounds(mesh=disk_mesh)
disk_heigh = disk_mesh_bound[1][2] - disk_mesh_bound[0][2]

disk_pose = Transform(np.array([0.6, 0.25, table_height + disk_mesh_bound[1][2]]))

benchmark_config = {4 : None}
scene_mngr = SceneManager("visual", is_pyplot=False, benchmark=benchmark_config)

for i in range(7):
    hanoi_disk_name = 'hanoi_disk_' + str(i+1)
    hanoi_mesh = get_object_mesh(f'hanoi_disk_{i}' + '.stl')
    scene_mngr.add_object(name=hanoi_disk_name, gtype="mesh", gparam=hanoi_mesh, h_mat=disk_pose.h_mat, color=[0., 1., 0.])

place_action = PlaceAction(scene_mngr, n_samples_held_obj=0, n_samples_support_obj=10)
surface_points_for_support_obj = list(place_action.get_surface_points_for_support_obj("hanoi_disk_1"))
fig, ax = p_utils.init_3d_figure(figsize=(10,6), dpi=120, name="Sampling Object")
# p_utils.plot_basis(ax)
place_action.scene_mngr.render_objects(ax, alpha=0.5)

for point, normal, (min_x, max_x, min_y, max_y) in surface_points_for_support_obj:
    if not (min_x <= point[0] <= max_x):
        continue
    if not (min_y <= point[1] <= max_y):
        continue
    place_action.scene_mngr.render.render_point(ax, point)

surface_points_for_support_obj = list(place_action.get_surface_points_for_support_obj("hanoi_disk_7"))
for point, normal, (min_x, max_x, min_y, max_y) in surface_points_for_support_obj:
    if not (min_x <= point[0] <= max_x):
        continue
    if not (min_y <= point[1] <= max_y):
        continue
    place_action.scene_mngr.render.render_point(ax, point)

place_action.show()