import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as p_utils

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

"""
8, 16
"""

table_mesh = get_object_mesh('ben_table.stl')
table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]
table_pose = Transform(pos=np.array([1.0, -0.4, -0.03]))
clearbox1_pose = Transform(pos=np.array([0.6, 0.25, table_height + 0.0607473]))
clearbox2_pose = Transform(pos=np.array([0.6, -0.25, table_height + 0.0607473]))

benchmark_config = {3 : None}
scene_mngr = SceneManager("collision", is_pyplot=False, benchmark=benchmark_config)

for i in range(20):
    clearbox_1_name = 'clearbox_1_' + str(i)
    clearbox_1_mesh = get_object_mesh(f'clearbox_{i}' + '.stl', scale=0.9)
    scene_mngr.add_object(name=clearbox_1_name, gtype="mesh", h_mat=clearbox1_pose.h_mat, gparam=clearbox_1_mesh, color=[0.8 + i*0.01, 0.8 + i*0.01, 0.8 + i*0.01])

    clearbox_2_name = 'clearbox_2_' + str(i)
    clearbox_2_mesh = get_object_mesh(f'clearbox_{i}' + '.stl', scale=0.9)
    scene_mngr.add_object(name=clearbox_2_name, gtype="mesh", h_mat=clearbox2_pose.h_mat, gparam=clearbox_2_mesh, color=[0.8 + i*0.01, 0.8 + i*0.01, 0.8 + i*0.01])

scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_robot(robot)

fig, ax = p_utils.init_3d_figure(name="Benchmark 3")
scene_mngr.render_scene(ax)
scene_mngr.show()