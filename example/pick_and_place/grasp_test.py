import numpy as np
import sys, os
import trimesh

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)


from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.tasks.activity import ActivityBase
from pykin.tasks.grasp import GraspManager
from pykin.utils.obstacle_utils import Obstacle
from pykin.utils.collision_utils import apply_robot_to_collision_manager
import pykin.utils.plot_utils as plt
from pykin.utils.transform_utils import get_pose_from_homogeneous, get_matrix_from_axis_angle
from pykin.utils.task_utils import normalize

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")

init_qpos = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk, geom="collision")
c_manager.filter_contact_names(robot, fk)

obs = Obstacle()
o_manager = CollisionManager()
obs_pos1 = Transform(pos=np.array([0.6, 0.2, 0.77]), rot=np.array([0, np.pi/2, np.pi/2]))
obs_pos2 = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, np.pi/2]))
obs_pos3 = Transform(pos=np.array([0.4, 0.24, 0.0]))

obj_mesh1 = trimesh.load(pykin_path+'/asset/objects/meshes/square_box.stl')
obj_mesh2 = trimesh.load(pykin_path+'/asset/objects/meshes/box_goal.stl')
obj_mesh3 = trimesh.load(pykin_path+'/asset/objects/meshes/custom_table.stl')

obj_mesh1.apply_scale(0.001)
obj_mesh2.apply_scale(0.001)
obj_mesh3.apply_scale(0.01)

obs(name="can", gtype="mesh", gparam=obj_mesh1, transform=obs_pos1)
obs(name="box", gtype="mesh", gparam=obj_mesh2, transform=obs_pos2)
obs(name="table", gtype="mesh", gparam=obj_mesh3, transform=obs_pos3)
o_manager.add_object("can", gtype="mesh", gparam=obj_mesh1, transform=obs_pos1.h_mat)
o_manager.add_object("box", gtype="mesh", gparam=obj_mesh2, transform=obs_pos2.h_mat)
o_manager.add_object("table", gtype="mesh", gparam=obj_mesh3, transform=obs_pos3.h_mat)

configures = {}
configures["gripper_names"] = ["right_gripper", "leftfinger", "rightfinger", "tcp"]
configures["gripper_max_width"] = 0.08
configures["gripper_max_depth"] = 0.035
configures["tcp_position"] = np.array([0, 0, 0.097])

#######################################
# 2. grasp test
grasp_man = GraspManager(
    robot, 
    c_manager, 
    o_manager, 
    mesh_path, 
    release_distance=0.01,
    retreat_distance=0.15,
    **configures)

######
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
# waypoints = grasp_man.get_grasp_waypoints(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)
# pre_grasp_pose = waypoints["pre_grasp"]
# grasp_pose = waypoints["grasp"]

# gripper = grasp_man.get_gripper_transformed(pre_grasp_pose, is_tcp=False)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_grasp_pose), axis=[1,1,1], scale=0.1)
# gripper = grasp_man.get_gripper_transformed(grasp_pose, is_tcp=False)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(grasp_pose), axis=[1,1,1], scale=0.1)
# plt.show_figure()
######

######
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
# eef_pose, tcp_pose, contact_points, _ = grasp_man.get_grasp_pose(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)
# grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
# gripper = grasp_man.get_gripper_transformed(tcp_pose)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
# plt.plot_line(ax, contact_points, 1)
# plt.show_figure()
######
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
# grasp_poses = list(grasp_man.generate_grasps(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10))

# grasp_pose, tcp_pose, contact_points, _ = grasp_man.filter_grasps(grasp_poses, 1)
# grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
# gripper = grasp_man.get_gripper_transformed(tcp_pose)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
# plt.plot_line(ax, contact_points, 1)
# plt.show_figure()
######

######
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
# grasp_poses = list(grasp_man.generate_grasps(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=2))

# for i, (eef_pose, tcp_pose, contact_points, _) in enumerate(grasp_poses):
#     plt.plot_vertices(ax, tcp_pose[:3, 3])
#     gripper = grasp_man.get_gripper_transformed(tcp_pose)
#     grasp_man.visualize_gripper(ax, gripper, alpha=1)
#     grasp_man.visualize_axis(ax, eef_pose, axis=[1,1,1], scale=0.05)
#     grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
#     plt.plot_line(ax, contact_points, 1)
# plt.show_figure()
######

####################################### 
# Release test
#########
# random sample
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
# for point, normal in grasp_man.generate_points_on_support(obj_mesh2, obs_pos2.h_mat, n_samples=2):
#     plt.plot_vertices(ax, point)
#     plt.plot_normal_vector(ax, point, normal, scale=0.1)

# for point, normal in grasp_man.generate_points_for_support(obj_mesh1, obs_pos1.h_mat, n_samples=10):
#     plt.plot_vertices(ax, point)
#     plt.plot_normal_vector(ax, point, normal, scale=0.1)
# plt.show_figure()
#########

#########
# eef_pose, tcp_pose, contact_points, normals = grasp_man.get_grasp_pose(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)

# for point_on_sup, normal_on_sup, point_for_sup, normal_for_sup in grasp_man.sample_supports(obj_mesh2, obs_pos2.h_mat, 1 , obj_mesh1, obs_pos1.h_mat, 6):
#     fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
#     plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
#     plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)

#     grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
#     gripper = grasp_man.get_gripper_transformed(tcp_pose)
#     grasp_man.visualize_gripper(ax, gripper, alpha=1)
#     grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
#     grasp_man.visualize_point(ax, tcp_pose)
    
#     plt.plot_vertices(ax, point_on_sup)
#     plt.plot_normal_vector(ax, point_on_sup, normal_on_sup, scale=0.1)
#     plt.plot_vertices(ax, point_for_sup)
#     plt.plot_normal_vector(ax, point_for_sup, normal_for_sup, scale=0.1)
#     plt.show_figure()
#########

# eef_pose, tcp_pose, contact_points, normals = grasp_man.get_grasp_pose(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)

# for point_on_sup, normal_on_sup, point_for_sup, normal_for_sup in grasp_man.sample_supports(obj_mesh2, obs_pos2.h_mat, 3, obj_mesh1, obs_pos1.h_mat, 3):
#     fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
#     normal_on_sup = -normal_on_sup
    
#     unit_vector_1 = normal_for_sup / np.linalg.norm(normal_for_sup)
#     unit_vector_2 = normal_on_sup / np.linalg.norm(normal_on_sup)

#     dot_product = np.dot(unit_vector_1, unit_vector_2)
#     angle = np.arccos(dot_product)

#     rot_axis = np.cross(unit_vector_2, unit_vector_1)
#     R = get_matrix_from_axis_angle(rot_axis, angle)

#     grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
#     gripper = grasp_man.get_gripper_transformed(tcp_pose)
#     grasp_man.visualize_gripper(ax, gripper, alpha=1)
#     # grasp_man.visualize_axis(ax, tcp_pose, axis=[1,1,1], scale=0.05)
#     # grasp_man.visualize_point(ax, tcp_pose)
    
#     A2B = np.eye(4)
#     A2B[:3, :3] = np.dot(R, obs_pos1.h_mat[:3, :3])
#     A2B[:3, 3] = obs_pos1.h_mat[:3, 3]

#     T = np.dot(obs_pos1.h_mat, np.linalg.inv(A2B))
#     gripper_pose = np.dot(T, tcp_pose)

#     gripper = grasp_man.get_gripper_transformed(gripper_pose)
#     grasp_man.visualize_gripper(ax, gripper, alpha=1)
#     # grasp_man.visualize_axis(ax, gripper_pose, axis=[1,1,1], scale=0.05)
#     # grasp_man.visualize_point(ax, gripper_pose)

#     plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=A2B, alpha=0.2, color='blue')
#     plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2, color='red')
#     plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
    
#     point_transformed = np.dot(point_for_sup - obs_pos1.pos, R) + obs_pos1.pos
#     normal_transformed = np.dot(normal_for_sup, R)

#     plt.plot_normal_vector(ax, point_transformed, normal_transformed, scale=0.3, edgecolor='green')
#     plt.plot_normal_vector(ax, point_on_sup, normal_on_sup, scale=0.1, edgecolor='green')
#     plt.plot_normal_vector(ax, point_for_sup, normal_for_sup, scale=0.1)

#     T = np.eye(4)
#     T[:3, :3] = A2B[:3, :3]
#     T[:3, 3] = obs_pos1.pos + (point_on_sup - point_transformed)

#     result_gripper_pose = np.eye(4)
#     result_gripper_pose[:3, :3] = gripper_pose[:3, :3]
#     result_gripper_pose[:3, 3] = gripper_pose[:3, 3] + (point_on_sup - point_transformed)
#     gripper = grasp_man.get_gripper_transformed(result_gripper_pose)
#     grasp_man.visualize_gripper(ax, gripper, alpha=1)
#     # grasp_man.visualize_axis(ax, result_gripper_pose, axis=[1,1,1], scale=0.05)
#     # grasp_man.visualize_point(ax, result_gripper_pose)


#     plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=T, alpha=0.2, color='blue')
#     plt.show_figure()

#########
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
# waypoints = grasp_man.get_grasp_waypoints(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)
# pre_grasp_pose = waypoints["pre_grasp"]
# grasp_pose = waypoints["grasp"]

# gripper = grasp_man.get_gripper_transformed(pre_grasp_pose, is_tcp=False)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_grasp_pose), axis=[1,1,1], scale=0.1)
# gripper = grasp_man.get_gripper_transformed(grasp_pose, is_tcp=False)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(grasp_pose), axis=[1,1,1], scale=0.1)

# support_poses = grasp_man.generate_supports(obj_mesh2, obs_pos2.h_mat, 10, obj_mesh1, obs_pos1.h_mat, 10)
# release_pose, result_obj_pose = grasp_man.filter_supports(support_poses)
# plt.plot_vertices(ax, grasp_man.obj_center_point)
# plt.plot_vertices(ax, grasp_man.obj_support_point)
# gripper = grasp_man.get_gripper_transformed(release_pose, is_tcp=False)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# gripper = grasp_man.get_gripper_transformed(grasp_man.pre_release_pose, is_tcp=False)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# gripper = grasp_man.get_gripper_transformed(grasp_man.tcp_pose)
# grasp_man.visualize_gripper(ax, gripper, alpha=1)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=result_obj_pose, alpha=0.2, color='blue')

# plt.show_figure()
########
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obs_pos1.h_mat, alpha=0.2)
plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
waypoints = grasp_man.get_grasp_waypoints(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)
pre_grasp_pose = waypoints["pre_grasp"]
grasp_pose = waypoints["grasp"]

gripper = grasp_man.get_gripper_transformed(pre_grasp_pose, is_tcp=False)
grasp_man.visualize_gripper(ax, gripper, alpha=1)
grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_grasp_pose), axis=[1,1,1], scale=0.1)
gripper = grasp_man.get_gripper_transformed(grasp_pose, is_tcp=False)
grasp_man.visualize_gripper(ax, gripper, alpha=1)
grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(grasp_pose), axis=[1,1,1], scale=0.1)

waypoints = grasp_man.get_release_waypoints(obj_mesh2, obs_pos2.h_mat, 10, obj_mesh1, obs_pos1.h_mat, 10, n_trials=1)
pre_release_pose = waypoints["pre_release"]
release_pose = waypoints["release"]

plt.plot_vertices(ax, grasp_man.obj_center_point)
plt.plot_vertices(ax, grasp_man.obj_support_point)
gripper = grasp_man.get_gripper_transformed(release_pose, is_tcp=False)

grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(release_pose), axis=[1,1,1], scale=0.1)

gripper = grasp_man.get_gripper_transformed(grasp_man.pre_release_pose, is_tcp=False)
grasp_man.visualize_gripper(ax, gripper, alpha=1)
# grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_release_pose), axis=[1,1,1], scale=0.1)

plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=grasp_man.result_obj_pose, alpha=0.2, color='blue')
plt.show_figure()