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
blue_box_pose = Transform(pos=np.array([0.6, 0.35, 0.77]))
green_box_pose = Transform(pos=np.array([0.6, 0.05, 0.77]))
# blue_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.06]))
# green_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.12]))
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

scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["red_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["blue_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["green_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.state.holding : None}
scene_mngr.update_logical_states()

pick = PickAction(scene_mngr, n_contacts=5, n_directions=10)

################## Action Test ##################
# actions = list(pick.get_possible_actions(level=0))
# # fig, ax = plt.init_3d_figure( name="all possible actions")
# for pick_actions in actions:
#     for grasp_pose in pick_actions[pick.action_info.GRASP_POSES]:
#         fig, ax = plt.init_3d_figure( name="all possible actions")
#         pick.render_axis(ax, grasp_pose)
#         pick.scene_mngr.render_objects(ax)
#         plt.plot_basis(ax)
#         pick.show()

################## Transitions Test ##################
actions = list(pick.get_possible_actions_level_1())
for action in actions:
    for scene in pick.get_possible_transitions(scene_mngr.scene, action=action):
        fig, ax = plt.init_3d_figure( name="all possible transitions")
        pick.scene_mngr.render_gripper(ax, scene, alpha=0.9, only_visible_axis=False)
        pick.scene_mngr.render_objects(ax)
        scene.show_logical_states()
        pick.scene_mngr.show()

# actions = list(pick.get_possible_actions(level=2))
# fig, ax = plt.init_3d_figure( name="all possible transitions")
# for action in actions:
#     for scene in pick.get_possible_transitions(scene_mngr.scene, action=action):
#         pick.scene_mngr.render_gripper(ax, scene, only_visible_axis=True)
# pick.scene_mngr.render_objects(ax)


################## Get grasp pose Test ##################
# fig, ax = plt.init_3d_figure( name="Get contact points")
# for obj in pick.scene_mngr.scene.objs:
#     if obj == "table":
#         continue
#     # contact_points = list(pick.get_contact_points(obj_name=obj))
#     # pick.render_points(ax, contact_points)

#     grasp_poses = pick.get_grasp_poses(obj)
#     # for grasp_pose in grasp_poses:
#     #     pick.render_axis(ax, grasp_pose)

#     grasp_poses_for_only_gripper = pick.get_grasp_poses_for_only_gripper(grasp_poses)
#     # for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
#     #     pick.render_axis(ax, grasp_pose_for_only_gripper, scale=0.05)

#     if not grasp_poses_for_only_gripper:
#         continue

#     goal_grasp_poses = list(pick.get_grasp_poses_for_robot(grasp_poses_for_only_gripper))
#     print(obj, len(goal_grasp_poses))
#     for grasp_pose in goal_grasp_poses:
#         pick.render_axis(ax, grasp_pose, scale=0.05)

# pick.scene_mngr.render_objects(ax)
# plt.plot_basis(ax)
# pick.show()