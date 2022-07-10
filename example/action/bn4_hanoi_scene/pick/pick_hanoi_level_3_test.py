import numpy as np
import sys, os

pykin_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import get_object_mesh, get_mesh_bounds
from pykin.action.pick import PickAction
import pykin.utils.plot_utils as p_utils

file_path = '../../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

table_mesh = get_object_mesh('ben_table.stl')
cylinder_mesh = get_object_mesh('hanoi_cylinder.stl', scale=[0.9, 0.9, 1.0])
disk_mesh = get_object_mesh('hanoi_disk.stl')

cylinder_mesh_bound = get_mesh_bounds(mesh=cylinder_mesh)
disk_mesh_bound = get_mesh_bounds(mesh=disk_mesh)
disk_heigh = disk_mesh_bound[1][2] - disk_mesh_bound[0][2]
table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]

table_pose = Transform(pos=np.array([1.0, -0.4, -0.03]))
cylinder1_pose = Transform(pos=np.array([0.6, -0.25, table_height + cylinder_mesh_bound[1][2]]))
cylinder2_pose = Transform(pos=np.array([0.6, 0, table_height + cylinder_mesh_bound[1][2]]))
cylinder3_pose = Transform(pos=np.array([0.6, 0.25, table_height + cylinder_mesh_bound[1][2]]))

disk_num = 3
disk_pose = [ Transform() for _ in range(disk_num)]
disk_object = [ 0 for _ in range(disk_num)]

benchmark_config = {4 : None}
scene_mngr = SceneManager("visual", is_pyplot=True, benchmark=benchmark_config)

theta = np.linspace(-np.pi, np.pi, disk_num)
for i in range(disk_num):
    for j in range(7):
        disk_pos = np.array([0.6, 0.25, table_height + disk_mesh_bound[1][2] + disk_heigh *i ])
        disk_ori = Transform._to_quaternion([0, 0, theta[i]])
        disk_pose[i] = Transform(pos=disk_pos, rot=disk_ori)
        disk_name = "hanoi_disk_" + str(i) + "_" + str(j)
        hanoi_mesh = get_object_mesh(f'hanoi_disk_{j}' + '.stl')
        scene_mngr.add_object(name=disk_name, gtype="mesh", gparam=hanoi_mesh, h_mat=disk_pose[i].h_mat, color=[0., 1., 0.])

scene_mngr.add_object(name="cylinder_1", gtype="mesh", gparam=cylinder_mesh, h_mat=cylinder1_pose.h_mat, color=[1, 0., 0.])
scene_mngr.add_object(name="cylinder_2", gtype="mesh", gparam=cylinder_mesh, h_mat=cylinder2_pose.h_mat, color=[1, 0., 0.])
scene_mngr.add_object(name="cylinder_3", gtype="mesh", gparam=cylinder_mesh, h_mat=cylinder3_pose.h_mat, color=[1, 0., 0.])
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_robot(robot)

scene_mngr.scene.logical_states["cylinder_1"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["cylinder_2"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["cylinder_3"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["hanoi_disk_0_6"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["hanoi_disk_1_6"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["hanoi_disk_0_6"]}
scene_mngr.scene.logical_states["hanoi_disk_2_6"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["hanoi_disk_1_6"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.logical_state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.logical_state.holding : None}
scene_mngr.update_logical_states()
scene_mngr.show_logical_states()

pick = PickAction(scene_mngr, n_contacts=0, n_directions=0)

################# Action Test ##################
actions = list(pick.get_possible_actions_level_1())

pick_joint_all_path = []
pick_all_objects = []
pick_all_object_poses = []

success_joint_path = False
for pick_action in actions:
    for idx, pick_scene in enumerate(pick.get_possible_transitions(scene_mngr.scene, action=pick_action)):
        # ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=pick_scene.grasp_poses)
        # if ik_solve:
        pick_joint_path = pick.get_possible_joint_path_level_3(scene=pick_scene, grasp_poses=pick_scene.grasp_poses)
        if pick_joint_path:
            success_joint_path = True
            pick_joint_all_path.append(pick_joint_path)
            pick_all_objects.append(pick.scene_mngr.attached_obj_name)
            pick_all_object_poses.append(pick.scene_mngr.scene.robot.gripper.pick_obj_pose)
        if success_joint_path: 
            break
print(len(pick_joint_all_path))
grasp_task_idx = 0
post_grasp_task_idx = 0
attach_idx = 0

for step, (all_joint_pathes, pick_object, pick_object_pose) in enumerate(zip(pick_joint_all_path, pick_all_objects, pick_all_object_poses)):
    for all_joint_path in all_joint_pathes:
        cnt = 0
        result_joint = []
        eef_poses = []
        fig, ax = p_utils.init_3d_figure( name="Level wise 3")
        for j, (task, joint_path) in enumerate(all_joint_path.items()):
            for k, joint in enumerate(joint_path):
                cnt += 1
                
                if task == "grasp":
                    grasp_task_idx = cnt
                if task == "post_grasp":
                    post_grasp_task_idx = cnt
                    
                if post_grasp_task_idx - grasp_task_idx == 1:
                    attach_idx = grasp_task_idx

                result_joint.append(joint)
                fk = pick.scene_mngr.scene.robot.forward_kin(joint)
                eef_poses.append(fk[pick.scene_mngr.scene.robot.eef_name].pos)

pick.scene_mngr.animation(
    ax,
    fig,
    joint_path=result_joint,
    eef_poses=eef_poses,
    visible_gripper=True,
    visible_text=True,
    alpha=1.0,
    interval=50,
    repeat=False,
    pick_object = [pick_all_objects[0]],
    attach_idx = [attach_idx],
    detach_idx = [])