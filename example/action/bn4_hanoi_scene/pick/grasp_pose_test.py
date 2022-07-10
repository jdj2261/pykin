import numpy as np
import sys, os
from copy import deepcopy


pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import get_object_mesh, get_mesh_bounds
from pykin.action.pick import PickAction
from pykin.scene.object import Object
import pykin.utils.plot_utils as p_utils

file_path = '../../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

table_mesh = get_object_mesh('ben_table.stl')
disk_mesh = get_object_mesh('hanoi_disk.stl')

disk_mesh_bound = get_mesh_bounds(mesh=disk_mesh)
disk_heigh = disk_mesh_bound[1][2] - disk_mesh_bound[0][2]
table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]

table_pose = Transform(pos=np.array([1.0, -0.4, -0.03]))

disk_num = 3
disk_pose = [ Transform() for _ in range(disk_num)]
disk_object = [ 0 for _ in range(disk_num)]

benchmark_config = {4 : None}
scene_mngr = SceneManager("collision", is_pyplot=True, benchmark=benchmark_config)

rot = [0, 0, np.pi/3]
disk_name = "hanoi_disk_6"
hanoi_mesh = get_object_mesh(disk_name + '.stl')
scene_mngr.add_object(name=disk_name, gtype="mesh", gparam=hanoi_mesh, h_mat=Transform(rot=rot).h_mat, color=[0., 1., 0.])
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_robot(robot)

fig, ax = p_utils.init_3d_figure(name="Level wise 2", visible_axis=True)

disk:Object = scene_mngr.scene.objs["hanoi_disk_6"]
disk_pose = disk.h_mat
disk_mesh = disk.gparam

copied_mesh = deepcopy(scene_mngr.scene.objs["hanoi_disk_6"].gparam)
copied_mesh.apply_transform(scene_mngr.scene.objs["hanoi_disk_6"].h_mat)

print(scene_mngr.scene.objs["hanoi_disk_6"].h_mat)
center_point = copied_mesh.center_mass

test = np.eye(4)
test[:3, :3] = scene_mngr.scene.objs["hanoi_disk_6"].h_mat[:3, :3]
test[:3, 3] = center_point

tcp_pose = np.eye(4)
# for theta in np.linspace(np.pi + np.pi/24, np.pi/12 + np.pi, 3):
for theta in np.linspace(0, np.pi * 2, 10):
    tcp_pose[:3,0] = [np.cos(theta), 0, np.sin(theta)]
    tcp_pose[:3,1] = [0, 1, 0]
    tcp_pose[:3,2] = [-np.sin(theta), 0, np.cos(theta)]
    # tcp_pose[:3,3] = center_point

    test_pose = np.dot(test, tcp_pose)

    grasp_pose = scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(test_pose)
    scene_mngr.render.render_axis(ax, grasp_pose)

scene_mngr.render.render_axis(ax, test)
scene_mngr.render.render_object(ax, disk)

p_utils.plot_basis(ax)
scene_mngr.show() 