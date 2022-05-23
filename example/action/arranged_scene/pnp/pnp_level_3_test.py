from cmath import pi
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

pick_actions = list(pick.get_possible_actions_level_1())


pick_joint_all_path = []
pick_all_objects = []
pick_all_object_poses = []


place_joint_all_path = []
place_all_objects = []
pick_all_object_poses = []
place_all_object_poses = []

success_joint_path = False
cnt = 0
for pick_action in pick_actions:
    for pick_scene in pick.get_possible_transitions(scene_mngr.scene, action=pick_action):
        ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=pick_scene.grasp_poses)
        if ik_solve:
            pick_joint_path = pick.get_possible_joint_path_level_3(scene=pick_scene, grasp_poses=grasp_pose)
            if pick_joint_path:
                pick_joint_all_path.append(pick_joint_path)
                pick_all_objects.append(pick.scene_mngr.attached_obj_name)
                pick_all_object_poses.append(pick.scene_mngr.scene.robot.gripper.pick_obj_pose)

                place_actions = list(place.get_possible_actions_level_1(pick_scene)) 
                for place_action in place_actions:
                    for place_scene in place.get_possible_transitions(scene=pick_scene, action=place_action):
                        ik_solve, release_poses = place.get_possible_ik_solve_level_2(scene=place_scene, release_poses=place_scene.release_poses)
                        if ik_solve:
                            place_joint_path = place.get_possible_joint_path_level_3(
                                scene=place_scene, release_poses=release_poses, init_thetas=pick_joint_path[-1][place.move_data.MOVE_default_grasp][-1])
                            if place_joint_path:
                                success_joint_path = True
                                place_joint_all_path.append(place_joint_path)
                                place_all_objects.append(place_scene.pick_obj_name)
                                pick_all_object_poses.append(pick_scene.pick_obj_default_pose)
                                place_all_object_poses.append(place_scene.objs[place_scene.pick_obj_name].h_mat)
                                break
                    if success_joint_path: 
                        break
        if success_joint_path: 
            break
    if success_joint_path: 
        break

result_path = pick_joint_all_path + place_joint_all_path
print(len(result_path))

fig, ax = plt.init_3d_figure( name="Level wise 3")
result_joint = []
eef_poses = []
cnt = 0

grasp_task_idx = 0
post_grasp_task_idx = 0
attach_idx = 0

release_task_idx = 0
post_release_task_idx = 0
detach_idx = 0

for all_joint_pathes in result_path:
    for all_joint_path in all_joint_pathes:
        for j, (task, joint_path) in enumerate(all_joint_path.items()):
            for k, joint in enumerate(joint_path):
                cnt += 1
                
                if task == "grasp":
                    grasp_task_idx = cnt
                if task == "post_grasp":
                    post_grasp_task_idx = cnt
                    
                if post_grasp_task_idx - grasp_task_idx == 1:
                    attach_idx = grasp_task_idx

                if task == "release":
                    release_task_idx = cnt
                if task == "post_release":
                    post_release_task_idx = cnt
                if post_release_task_idx - release_task_idx == 1:
                    detach_idx = release_task_idx

                result_joint.append(joint)
                fk = pick.scene_mngr.scene.robot.forward_kin(joint)
                eef_poses.append(fk[place.scene_mngr.scene.robot.eef_name].pos)

pick.scene_mngr.animation(
    ax,
    fig,
    joint_path=result_joint,
    eef_poses=eef_poses,
    visible_gripper=True,
    visible_text=True,
    alpha=1.0,
    interval=1,
    repeat=False,
    pick_object = [pick_all_objects[0]],
    attach_idx = [attach_idx],
    detach_idx = [detach_idx])