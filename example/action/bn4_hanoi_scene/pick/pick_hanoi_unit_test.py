import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
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
scene_mngr = SceneManager("collision", is_pyplot=True, benchmark=benchmark_config)

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

pick = PickAction(scene_mngr, n_contacts=5, n_directions=100)

###### All Contact Points #######
fig, ax = p_utils.init_3d_figure(name="Get contact points")
contact_points = pick.get_contact_points(obj_name="hanoi_disk_2_6")
pick.scene_mngr.render_objects(ax, alpha=0.3)
p_utils.plot_basis(ax)
for contact_point in contact_points:
    contact_point_1, contact_point_2 = contact_point[0], contact_point[1]

    if not pick.scene_mngr.is_pyplot:
        pick.scene_mngr.render.render_point(ax, contact_point_1, radius=0.001)
        pick.scene_mngr.render.render_point(ax, contact_point_2, radius=0.001)
    else:
        pick.scene_mngr.render.render_point(ax, contact_point_1)
        pick.scene_mngr.render.render_point(ax, contact_point_2)
# pick.show()

##### All Grasp Pose #######
grasp_poses = list(pick.get_all_grasp_poses("hanoi_disk_2_6"))
fig, ax = p_utils.init_3d_figure(name="Get Grasp Pose")
pick.scene_mngr.render_objects(ax)
p_utils.plot_basis(ax)
for grasp_pose in grasp_poses:
    pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_grasp])
    
# ###### Level wise - 1 #######
fig, ax = p_utils.init_3d_figure(name="Level wise 1")
pick.scene_mngr.render_objects(ax)
p_utils.plot_basis(ax)
grasp_poses_for_only_gripper = list(pick.get_all_grasp_poses_not_collision(grasp_poses))
for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
    pick.scene_mngr.render.render_axis(ax, grasp_pose_for_only_gripper[pick.move_data.MOVE_grasp])
    pick.scene_mngr.render.render_axis(ax, grasp_pose_for_only_gripper[pick.move_data.MOVE_pre_grasp])
    pick.scene_mngr.render.render_axis(ax, grasp_pose_for_only_gripper[pick.move_data.MOVE_post_grasp])
    pick.scene_mngr.render_gripper(ax, alpha=0.7, pose=grasp_pose_for_only_gripper[pick.move_data.MOVE_grasp])

####### Level wise - 2 #######
fig, ax = p_utils.init_3d_figure(name="Level wise 2")
pick.scene_mngr.render_scene(ax)
p_utils.plot_basis(ax)
for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
    thetas, grasp_pose = pick.compute_ik_solve_for_robot(grasp_pose=grasp_pose_for_only_gripper)
    if grasp_pose:
        pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_grasp])

pick.show()