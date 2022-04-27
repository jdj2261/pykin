import numpy as np
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.utils.transform_utils import get_matrix_from_rpy
from pykin.utils.kin_utils import ShellColors as sc
import pykin.utils.plot_utils as plt

file_path = '../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")

file_path = '../../../asset/urdf/panda/panda.urdf'
panda_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[0, 0, 0]))
custom_fpath = '../../../asset/config/panda_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.06]))
green_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.12]))
support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

red_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
blue_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
green_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
box_goal_mesh = get_object_mesh('box_goal.stl', 0.001)
table_mesh = get_object_mesh('custom_table.stl', 0.01)

scene_mngr = SceneManager("visual", is_pyplot=False)
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="red_box", gtype="mesh", gparam=red_cube_mesh, h_mat=red_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="blue_box", gtype="mesh", gparam=blue_cube_mesh, h_mat=blue_box_pose.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="green_box", gtype="mesh", gparam=green_cube_mesh, h_mat=green_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=box_goal_mesh, h_mat=support_box_pose.h_mat, color=[1.0, 0, 1.0])
scene_mngr.add_robot(robot, init_qpos)

grasp_pose = green_box_pose.h_mat
r_mat = get_matrix_from_rpy(np.array([0, np.pi/2, 0]))
grasp_pose[:3, :3] = r_mat
grasp_pose[:3, 3] = grasp_pose[:3, 3] - [0.1, 0, 0]
target_thetas = scene_mngr.compute_ik(grasp_pose)
############################# Object Attach to Robot Test #############################
print(f"{sc.OKBLUE}Move grasp pose{sc.ENDC}")
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Move grasp pose")
scene_mngr.set_robot_eef_pose(target_thetas)
scene_mngr.obj_collision_mngr.show_collision_info("Object")
scene_mngr.robot_collision_mngr.show_collision_info("Robot")
scene_mngr.render_scene(ax, visible_geom=True, alpha=0.7)

print(f"{sc.OKBLUE}Attach Object to Robot{sc.ENDC}")
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Attach Object")
scene_mngr.attach_object_on_gripper("green_box", False)
scene_mngr.obj_collision_mngr.show_collision_info("Object")
scene_mngr.robot_collision_mngr.show_collision_info("Robot")
scene_mngr.render_scene(ax, visible_geom=True, alpha=0.7)

print(f"{sc.OKBLUE}Detach Object{sc.ENDC}")
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Detach Object")
scene_mngr.detach_object_from_gripper(False)
scene_mngr.set_robot_eef_pose(init_qpos)
scene_mngr.obj_collision_mngr.show_collision_info("Object")
scene_mngr.robot_collision_mngr.show_collision_info("Robot")
scene_mngr.render_scene(ax, visible_geom=True, alpha=0.7)
scene_mngr.show()

# # ############################ Object Attach to Gripper Test #############################
# scene_mngr.add_object(name="green_box", gtype="mesh", gparam=green_cube_mesh, h_mat=green_box_pose.h_mat, color=[0, 1, 0])

# print(f"{sc.OKBLUE}Move only gripper{sc.ENDC}")
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Move only gripper")
# scene_mngr.set_gripper_pose(grasp_pose)
# scene_mngr.obj_collision_mngr.show_collision_info("Object")
# scene_mngr.robot_collision_mngr.show_collision_info("Robot")
# scene_mngr.gripper_collision_mngr.show_collision_info("Gripper")
# scene_mngr.render_objects_and_gripper(ax, alpha=0.8)

# print(f"{sc.OKBLUE}Attach object only gripper{sc.ENDC}")
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Attach object only gripper")
# scene_mngr.attach_object_on_gripper("green_box", only_gripper=True)
# scene_mngr.obj_collision_mngr.show_collision_info("Object")
# scene_mngr.robot_collision_mngr.show_collision_info("Robot")
# scene_mngr.gripper_collision_mngr.show_collision_info("Gripper")
# scene_mngr.render_objects_and_gripper(ax, alpha=0.8)

# print(f"{sc.OKBLUE}Detach object only gripper{sc.ENDC}")
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Detach object only gripper")
# scene_mngr.detach_object_from_gripper(only_gripper=True)
# scene_mngr.obj_collision_mngr.show_collision_info("Object")
# scene_mngr.robot_collision_mngr.show_collision_info("Robot")
# scene_mngr.gripper_collision_mngr.show_collision_info("Gripper")
# scene_mngr.render_objects_and_gripper(ax, alpha=0.8)
# scene_mngr.show()