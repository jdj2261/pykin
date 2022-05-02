import numpy as np
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.pick import PickAction
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

scene_mngr = SceneManager("collision", is_pyplot=True)
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="red_box", gtype="mesh", gparam=red_cube_mesh, h_mat=red_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="blue_box", gtype="mesh", gparam=blue_cube_mesh, h_mat=blue_box_pose.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="green_box", gtype="mesh", gparam=green_cube_mesh, h_mat=green_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=box_goal_mesh, h_mat=support_box_pose.h_mat, color=[1.0, 0, 1.0])
scene_mngr.add_robot(robot, init_qpos)

## Init logical states
scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["red_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["blue_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["green_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.state.holding : None}
scene_mngr.update_logical_states()

pick = PickAction(scene_mngr, 10, 50)
# pick_actions = list(pick.get_possible_actions_level_1(scene_mngr.scene))
# # fig, ax = plt.init_3d_figure( name="all possible actions")
# for pick_action in pick_actions:
#     for grasp_pose in pick_action[pick.action_info.GRASP_POSES]:
#         fig, ax = plt.init_3d_figure( name="all possible actions")
#         pick.render_axis(ax, grasp_pose)
#         pick.scene_mngr.render_objects(ax)
#         plt.plot_basis(ax)
#         pick.show()



####### All Contact Points #######
# fig, ax = plt.init_3d_figure(name="Get contact points")
# contact_points = list(pick.get_contact_points(obj_name="green_box"))
# pick.render_points(ax, contact_points)
# pick.scene_mngr.render_objects(ax, alpha=0.5)
# plt.plot_basis(ax)

###### All Grasp Pose #######
grasp_poses = list(pick.get_all_grasp_poses("green_box"))
fig, ax = plt.init_3d_figure(name="Get Grasp Pose")
for grasp_pose in grasp_poses:
    # pick.render_axis(ax, grasp_pose[pick.grasp_name.PRE_GRASP])
    pick.render_axis(ax, grasp_pose[pick.grasp_name.GRASP])
    # pick.render_axis(ax, grasp_pose[pick.grasp_name.POST_GRASP])
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
# pick.show()

# ###### Level wise - 1 #######
fig, ax = plt.init_3d_figure(name="Level wise 1")
grasp_poses_for_only_gripper = list(pick.get_all_grasp_poses_for_only_gripper(grasp_poses))
for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
    # pick.render_axis(ax, grasp_pose_for_only_gripper[pick.grasp_name.PRE_GRASP], scale=0.05)
    pick.render_axis(ax, grasp_pose_for_only_gripper[pick.grasp_name.GRASP], scale=0.05)
    # pick.render_axis(ax, grasp_pose_for_only_gripper[pick.grasp_name.POST_GRASP], scale=0.01)
    pick.scene_mngr.render_gripper(ax, alpha=0.3, robot_color='b', pose=grasp_pose_for_only_gripper[pick.grasp_name.GRASP])
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
# pick.show()

####### Level wise - 2 #######
fig, ax = plt.init_3d_figure(name="Level wise 2")
for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
    if pick.check_possible_action_level_2(grasp_pose=grasp_pose_for_only_gripper):
        pick.render_axis(ax, grasp_pose_for_only_gripper, scale=0.05)
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
pick.show()
