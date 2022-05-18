import numpy as np
import sys, os
import pprint

pykin_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.pick import PickAction
from pykin.action.place import PlaceAction
import pykin.utils.plot_utils as plt

file_path = '../../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

file_path = '../../../../asset/urdf/panda/panda.urdf'
panda_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[0, 0, 0]))

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.35, 0.77]))
green_box_pose = Transform(pos=np.array([0.6, 0.05, 0.77]))
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
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.set_logical_state("goal_box", ("on", "table"))
scene_mngr.set_logical_state("red_box", ("on", "table"))
scene_mngr.set_logical_state("blue_box", ("on", "table"))
scene_mngr.set_logical_state("green_box", ("on", "table"))
scene_mngr.set_logical_state("table", ("static", True))
scene_mngr.set_logical_state(scene_mngr.gripper_name, ("holding", None))
scene_mngr.update_logical_states(init=True)

pick = PickAction(scene_mngr, n_contacts=10, n_directions=10)
place = PlaceAction(scene_mngr, n_samples_held_obj=3, n_samples_support_obj=3)

pick_joint_all_path = []
# pick
# step 1. action[type] == pick
pick_action = pick.get_action_level_1_for_single_object(scene_mngr.scene, "green_box")

for pick_scene in pick.get_possible_transitions(scene_mngr.scene, pick_action):
    ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=pick_scene.grasp_poses)
    if ik_solve:
        pick_joint_path = pick.get_possible_joint_path_level_3(scene=pick_scene, grasp_poses=grasp_pose)
        if pick_joint_path:
            pick_joint_all_path.append(pick_joint_path)

            place_action = place.get_action_level_1_for_single_object(pick_scene, "goal_box")
            for place_scene in place.get_possible_transitions(scene=pick_scene, action=place_action):
                ik_solve, release_poses = place.get_possible_ik_solve_level_2(scene=place_scene, release_poses=place_scene.release_poses):
                if ik_sovle:
                    

pprint.pprint(pick_joint_path)

    # fig, ax = plt.init_3d_figure( name="all possible transitions")
    # pick.scene_mngr.render_gripper(ax, pick_scene, alpha=0.9, only_visible_axis=False)
    # pick.scene_mngr.render_objects(ax, pick_scene)
    # pick_scene.show_logical_states()
    # pick.scene_mngr.show()

    # print(pick_scene.grasp_poses)

# grasp_poses = pick_action[pick.action_info.GRASP_POSES]
# for grasp_pose in grasp_poses:
    # print(grasp_pose[pick.move_data.MOVE_grasp])


# step 2. transition
# step 3. level-wise 3
# step 4. get joint path for pick

# pick
# step 1. action[type] == place
# step 2. transition
# step 3. level-wise 3
# step 4. get joint path for place