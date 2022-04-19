import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

file_path = '../../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name("sawyer_base", "sawyer_right_hand")

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.06]))
green_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.12]))
support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
box_goal_mesh = get_object_mesh('box_goal.stl', 0.001)
table_mesh = get_object_mesh('custom_table.stl', 0.01)

scene_mngr = SceneManager()
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="red_box", gtype="mesh", gparam=cube_mesh, h_mat=red_box_pose.h_mat, color=[1, 0, 0])
scene_mngr.add_object(name="blue_box", gtype="mesh", gparam=cube_mesh, h_mat=blue_box_pose.h_mat, color=[0, 0, 1])
scene_mngr.add_object(name="green_box", gtype="mesh", gparam=cube_mesh, h_mat=green_box_pose.h_mat, color=[0, 1, 0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=box_goal_mesh, h_mat=support_box_pose.h_mat, color=[1, 0, 1])
scene_mngr.add_robot(robot)

############################# Render Test #############################
target_thetas = np.array([0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8, 0.0, 0])
scene_mngr.set_robot_eef_pose(target_thetas)

scene_mngr.render_all_scene(ax, robot_color='b')
plt.show_figure()


