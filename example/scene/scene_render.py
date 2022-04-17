import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name("panda_link_0", "panda_right_hand")

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

scene_mngr.logical_states["red_box"] = {scene_mngr.state.on : scene_mngr.objs["table"]}
scene_mngr.logical_states["blue_box"] = {scene_mngr.state.on : scene_mngr.objs["red_box"]}
scene_mngr.logical_states["green_box"] = {scene_mngr.state.on : scene_mngr.objs["blue_box"]}
scene_mngr.logical_states["goal_box"] = {scene_mngr.state.on : scene_mngr.objs["table"]}
scene_mngr.logical_states["table"] = {scene_mngr.state.static : True}
scene_mngr.logical_states[scene_mngr.gripper_name] = {scene_mngr.state.holding : None}

scene_mngr.render_all_scene(ax, robot_color='b')
# scene_mngr.render_object_and_gripper(ax, gripper_color='b')
# scene_mngr.render_object(ax)
# scene_mngr.render_gripper(ax, gripper_color='b')

plt.show_figure()

