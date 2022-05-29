import numpy as np
import sys, os

pykin_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
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
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["red_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["blue_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["red_box"]}
scene_mngr.scene.logical_states["green_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["blue_box"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.logical_state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.logical_state.holding : None}
scene_mngr.update_logical_states()

pick = PickAction(scene_mngr, n_contacts=10, n_directions=10)
place = PlaceAction(scene_mngr, n_samples_held_obj=10, n_samples_support_obj=10)

pnp_joint_all_pathes = []
pick_all_objects = []
place_all_objects = []

place_all_object_poses = []
success_joint_path = False
# pick
# step 1. action[type] == pick
pick_action = pick.get_action_level_1_for_single_object(scene_mngr.scene, "green_box")

for pick_scene in pick.get_possible_transitions(scene_mngr.scene, pick_action):
    ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=pick_scene.grasp_poses)
    if ik_solve:
        pick_joint_path = pick.get_possible_joint_path_level_3(scene=pick_scene, grasp_poses=grasp_pose)
        if pick_joint_path:
            place_action = place.get_action_level_1_for_single_object("goal_box", "green_box", pick_scene.robot.gripper.grasp_pose, scene=pick_scene)
            for place_scene in place.get_possible_transitions(scene=pick_scene, action=place_action):
                ik_solve, release_poses = place.get_possible_ik_solve_level_2(scene=place_scene, release_poses=place_scene.release_poses)
                if ik_solve:
                    place_joint_path = place.get_possible_joint_path_level_3(
                        scene=place_scene, release_poses=release_poses, init_thetas=pick_joint_path[-1][place.move_data.MOVE_default_grasp][-1])
                    if place_joint_path:
                        pick_action_2 = pick.get_action_level_1_for_single_object(scene_mngr.scene, "blue_box")
                        for pick_scene_2 in pick.get_possible_transitions(place_scene, pick_action_2):
                            ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=pick_scene_2.grasp_poses)
                            if ik_solve:
                                pick_joint_path_2 = pick.get_possible_joint_path_level_3(
                                    scene=pick_scene_2, grasp_poses=grasp_pose, init_thetas=place_joint_path[-1][place.move_data.MOVE_default_release][-1])
                                if pick_joint_path_2:
                                    place_action2 = place.get_action_level_1_for_single_object("green_box", "blue_box", pick_scene_2.robot.gripper.grasp_pose, scene=pick_scene_2)
                                    for place_scene_2 in place.get_possible_transitions(scene=pick_scene_2, action=place_action2):
                                        ik_solve, release_poses = place.get_possible_ik_solve_level_2(scene=place_scene_2, release_poses=place_scene_2.release_poses)
                                        if ik_solve:
                                            place_joint_path2 = place.get_possible_joint_path_level_3(
                                                scene=place_scene_2, release_poses=release_poses, init_thetas=pick_joint_path_2[-1][place.move_data.MOVE_default_grasp][-1])
                                            if place_joint_path2:
                                                pick_action_3 = pick.get_action_level_1_for_single_object(scene_mngr.scene, "red_box")
                                                for pick_scene_3 in pick.get_possible_transitions(place_scene_2, pick_action_3):
                                                    ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=pick_scene_3.grasp_poses)
                                                    if ik_solve:
                                                        pick_joint_path_3 = pick.get_possible_joint_path_level_3(
                                                            scene=pick_scene_3, grasp_poses=grasp_pose, init_thetas=place_joint_path2[-1][place.move_data.MOVE_default_release][-1])
                                                        if pick_joint_path_3:
                                                            place_action3 = place.get_action_level_1_for_single_object("blue_box", "red_box", pick_scene_3.robot.gripper.grasp_pose, scene=pick_scene_3)
                                                            for place_scene_3 in place.get_possible_transitions(scene=pick_scene_3, action=place_action3):
                                                                ik_solve, release_poses = place.get_possible_ik_solve_level_2(scene=place_scene_3, release_poses=place_scene_3.release_poses)
                                                                if ik_solve:
                                                                    place_joint_path3 = place.get_possible_joint_path_level_3(
                                                                        scene=place_scene_2, release_poses=release_poses, init_thetas=pick_joint_path_3[-1][place.move_data.MOVE_default_grasp][-1])

                                                                    success_joint_path = True
                                                                    pnp_joint_all_pathes.append((pick_joint_path + place_joint_path + pick_joint_path_2 + place_joint_path2 + pick_joint_path_3 + place_joint_path3))
                                                                    pick_all_objects.append([pick_scene.robot.gripper.attached_obj_name, pick_scene_2.robot.gripper.attached_obj_name, pick_scene_3.robot.gripper.attached_obj_name])
                                                                    place_all_object_poses.append([place_scene.objs[place_scene.pick_obj_name].h_mat, place_scene_2.objs[place_scene_2.pick_obj_name].h_mat, place_scene_3.objs[place_scene_3.pick_obj_name].h_mat])
                                                                    break
                                                    if success_joint_path: 
                                                        break
                                        if success_joint_path: 
                                            break
                            if success_joint_path: 
                                break
                if success_joint_path: 
                    break
    if success_joint_path: 
        break

# pprint.pprint(pnp_joint_all_path)
# print(pnp_joint_all_pathes)

for pnp_joint_all_path, pick_all_object, place_all_object_pose in zip(pnp_joint_all_pathes, pick_all_objects, place_all_object_poses):
    # fig, ax = plt.init_3d_figure( name="Level wise 3")
    result_joint = []
    eef_poses = []
    attach_idxes = []
    detach_idxes = []

    attach_idx = 0
    detach_idx = 0

    grasp_task_idx = 0
    post_grasp_task_idx = 0

    release_task_idx = 0
    post_release_task_idx = 0
    cnt = 0
    for pnp_joint_path in pnp_joint_all_path:        
        for j, (task, joint_path) in enumerate(pnp_joint_path.items()):
            for k, joint in enumerate(joint_path):
                cnt += 1
                
                if task == pick.move_data.MOVE_grasp:
                    grasp_task_idx = cnt
                if task == pick.move_data.MOVE_post_grasp:
                    post_grasp_task_idx = cnt
                    
                if post_grasp_task_idx - grasp_task_idx == 1:
                    attach_idx = grasp_task_idx
                    attach_idxes.append(attach_idx)

                if task == place.move_data.MOVE_release:
                    release_task_idx = cnt
                if task == place.move_data.MOVE_post_release:
                    post_release_task_idx = cnt
                if post_release_task_idx - release_task_idx == 1:
                    detach_idx = release_task_idx
                    detach_idxes.append(detach_idx)
                
                result_joint.append(joint)
                fk = pick.scene_mngr.scene.robot.forward_kin(joint)
                eef_poses.append(fk[place.scene_mngr.scene.robot.eef_name].pos)

    fig, ax = plt.init_3d_figure( name="Level wise 3")
    pick.scene_mngr.animation(
        ax,
        fig,
        init_scene=pick_scene,
        joint_path=result_joint,
        eef_poses=None,
        visible_gripper=True,
        visible_text=True,
        alpha=1.0,
        interval=1,
        repeat=False,
        pick_object = pick_all_object,
        attach_idx = attach_idxes,
        detach_idx = detach_idxes,
        place_obj_pose= place_all_object_pose)