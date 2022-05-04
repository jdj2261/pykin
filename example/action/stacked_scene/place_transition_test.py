import numpy as np
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.place import PlaceAction
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
scene_mngr.update_logical_states(True)

pick = PickAction(scene_mngr, 3, 10)
place = PlaceAction(scene_mngr, n_samples_held_obj=3, n_samples_support_obj=3)

actions = list(pick.get_possible_actions_level_1())

## place action test
for action in actions:
    for idx, scene in enumerate(pick.get_possible_transitions(scene_mngr.scene, action=action)):
        place_actions = list(place.get_possible_actions_level_1(scene))
        for place_action in place_actions:
            for release_pose in place_action[place.action_info.RELEASE_POSES]:
                ik_solve = place.get_possible_ik_solve_level_2(scene, release_pose=release_pose)
                if ik_solve is not None:
                    fig, ax = plt.init_3d_figure( name="all possible transitions")
                    place.scene_mngr.set_gripper_pose(release_pose[place.release_name.RELEASE])
                    place.scene_mngr.render_gripper(ax, alpha=0.9, only_visible_axis=False)
                    place.scene_mngr.render_objects(ax)
                    place.scene_mngr.robot_collision_mngr.show_collision_info()
                    place.render_axis(ax, release_pose[place.release_name.RELEASE])
                    place.render_axis(ax, release_pose[place.release_name.PRE_RELEASE])
                    place.render_axis(ax, release_pose[place.release_name.POST_RELEASE])
                    pick.scene_mngr.show()
        place.scene_mngr.detach_object_from_gripper()

# place ik solve level 2
# for action in actions:
#     for idx, scene in enumerate(pick.get_possible_transitions(scene_mngr.scene, action=action)):
#         place_actions = list(place.get_possible_actions_level_1(scene))
#         for place_action in place_actions:
#             for release_pose in place_action[place.action_info.RELEASE_POSES]:
#                 ik_solve = place.get_possible_ik_solve_level_2(scene, release_pose=release_pose)
#                 if ik_solve is not None:
#                     fig, ax = plt.init_3d_figure( name="all possible transitions")
#                     place.scene_mngr.set_gripper_pose(release_pose[place.release_name.RELEASE])
#                     place.scene_mngr.render_gripper(ax, alpha=0.9, only_visible_axis=False)
#                     place.scene_mngr.render_objects(ax)
#                     place.scene_mngr.robot_collision_mngr.show_collision_info()
#                     place.render_axis(ax, release_pose[place.release_name.RELEASE])
#                     place.render_axis(ax, release_pose[place.release_name.PRE_RELEASE])
#                     place.render_axis(ax, release_pose[place.release_name.POST_RELEASE])
#                     pick.scene_mngr.show()
#         place.scene_mngr.detach_object_from_gripper()


## pick possible transitions
# for action in actions:
#     for idx, scene in enumerate(pick.get_possible_transitions(scene_mngr.scene, action=action)):
#         fig, ax = plt.init_3d_figure( name="all possible transitions")
#         pick.scene_mngr.render_gripper(ax, scene, alpha=0.9, only_visible_axis=False)
#         pick.scene_mngr.render_objects(ax)
#         pick.scene_mngr.show_logical_states()
#         # pick.scene_mngr.show_scene_info()
#         pick.scene_mngr.show_logical_states()
#         pick.scene_mngr.detach_object_from_gripper()
#         pick.scene_mngr.show()

# pick_actions = list(pick.get_possible_actions_level_1())
# # fig, ax = plt.init_3d_figure( name="Get Release Pose")
# for pick_action_level_1 in pick_actions:
#     for pick_scene in pick.get_possible_transitions(scene_mngr.scene, action=pick_action_level_1):
#         # fig, ax = plt.init_3d_figure( name="Get Release Pose")
#         # pick.scene_mngr.render_objects(ax)
#         # pick.scene_mngr.render_gripper(ax)
#         # pick.scene_mngr.detach_object_from_gripper()
#         # pick.show()
#         place_actions = list(place.get_possible_actions_level_1(pick_scene)) 
#         for place_action_level_1 in place_actions:
#             for place_scene in place.get_possible_transitions(pick_scene, action=place_action_level_1):
#                 fig, ax = plt.init_3d_figure( name="Get Release Pose")
#                 pick.scene_mngr.render_objects(ax)
#                 pick.scene_mngr.render_gripper(ax)
#                 pick.show()

        # place.scene_mngr.detach_object_from_gripper()
