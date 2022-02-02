import numpy as np
import sys, os
import trimesh

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.tasks.grasp import GraspManager, GraspStatus
from pykin.utils.object_utils import ObjectManager
import pykin.utils.plot_utils as plt

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")

init_qpos = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, fk, geom="collision")
c_manager.show_collision_info()

obs_pos1 = Transform(pos=np.array([0.6, 0.2, 0.77]), rot=np.array([0, np.pi/2, np.pi/2]))
obs_pos2 = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
obs_pos3 = Transform(pos=np.array([0.4, 0.24, 0.0]))

obj_mesh1 = trimesh.load(pykin_path+'/asset/objects/meshes/square_box.stl')
obj_mesh2 = trimesh.load(pykin_path+'/asset/objects/meshes/box_goal.stl')
obj_mesh3 = trimesh.load(pykin_path+'/asset/objects/meshes/custom_table.stl')

obj_mesh1.apply_scale(0.001)
obj_mesh2.apply_scale(0.001)
obj_mesh3.apply_scale(0.01)

# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_robot(robot, ax, fk, visible_collision=True)
# plt.plot_mesh(ax, obj_mesh1)
# plt.plot_mesh(ax, obj_mesh2)
# plt.plot_mesh(ax, obj_mesh3, h_mat=obs_pos3.h_mat)


objects = ObjectManager()
objects.add_object("obj_1", gtype="mesh", gparam=obj_mesh1, h_mat=obs_pos1.h_mat, for_grasp=True)
objects.add_object("box", gtype="mesh", gparam=obj_mesh2, h_mat=obs_pos2.h_mat, for_support=True)
objects.add_object(name="table", gtype="mesh", gparam=obj_mesh3, h_mat=obs_pos3.h_mat)

o_manager = CollisionManager()
o_manager.setup_object_collision(objects)

o_manager.show_collision_info()

# result, name = c_manager.in_collision_other(o_manager, True)
# print(result, name)
# plt.show_figure()

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
    retreat_distance=0.15,
    release_distance=0.01,
    **configures)

# ######
# # generate_grasps
# # fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)
# # grasp_poses = grasp_man.generate_grasps(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=1)

# # for i, (eef_pose, gripper) in enumerate(grasp_poses):
# #     grasp_man.visualize_gripper(ax, gripper, alpha=1)
# #     grasp_man.visualize_axis(ax, eef_pose, axis=[1,1,1], scale=0.05)
# # plt.show_figure()
# ######

# ######
# get grasp pose
# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.5, color='orange')
# plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
# plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)
# grasp_pose = grasp_man.get_grasp_pose(obj_mesh1, obs_pos1.h_mat, limit_angle=0.1, num_grasp=100, n_trials=1)
# grasp_man.visualize_axis(ax, grasp_man.tcp_pose, axis=[1,1,1], scale=0.05)
# gripper = grasp_man.get_transformed_gripper_fk(grasp_man.tcp_pose)
# grasp_man.visualize_gripper(ax, gripper, alpha=1.0,color='blue')
# grasp_man.visualize_axis(ax, grasp_man.tcp_pose, axis=[1,1,1], scale=0.05)
# plt.plot_line(ax, grasp_man.contact_points, 1)
# plt.show_figure()
# ######

# ######
# # # get grasp waypoints
# # fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.2, color='blue')
# # plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)

# # grasp_object_info = objects.get_info("obj_1")
# # waypoints = grasp_man.get_grasp_waypoints(grasp_object_info, limit_angle=0.1, num_grasp=10, n_trials=10)
# # pre_grasp_pose = waypoints[GraspStatus.pre_grasp_pose]
# # grasp_pose = waypoints[GraspStatus.grasp_pose]

# # gripper = grasp_man.get_transformed_gripper_fk(pre_grasp_pose, is_tcp=False)
# # grasp_man.visualize_gripper(ax, gripper, alpha=1)
# # grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_grasp_pose), axis=[1,1,1], scale=0.1)
# # gripper = grasp_man.get_transformed_gripper_fk(grasp_pose, is_tcp=False)
# # grasp_man.visualize_gripper(ax, gripper, alpha=1)
# # grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(grasp_pose), axis=[1,1,1], scale=0.1)
# # plt.show_figure()
# ######

# ####################################### 
# # Release test
# #########
# # random sample
# # fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)
# # for point, normal in grasp_man.generate_points_on_support(obj_mesh2, obs_pos2.h_mat, n_samples=10):
# #     plt.plot_vertices(ax, point)
# #     plt.plot_normal_vector(ax, point, normal, scale=0.1)

# # for point, normal in grasp_man.generate_points_for_support(obj_mesh1, obs_pos1.h_mat, n_samples=10):
# #     plt.plot_vertices(ax, point)
# #     plt.plot_normal_vector(ax, point, normal, scale=0.1)
# # plt.show_figure()
# #########
# # support_poses = grasp_man.generate_supports(obj_mesh2, obs_pos2.h_mat, 3, obj_mesh1, obs_pos1.h_mat, 10)

# # for _, result_obj_pose in support_poses:
# #     fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# #     plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.2, color='blue')
# #     plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
# #     plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)
# #     plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=grasp_man.obj_pose_transformed_for_sup, alpha=0.2, color='red')
# #     plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=result_obj_pose, alpha=0.2, color='red')

# #     plt.show_figure()


# # #########
# # fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
# # plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)
# # waypoints = grasp_man.get_grasp_waypoints(obj_mesh=obj_mesh1, obj_pose=obs_pos1.h_mat, limit_angle=0.1, num_grasp=10, n_trials=10)
# # pre_grasp_pose = waypoints[GraspStatus.pre_grasp_pose]
# # grasp_pose = waypoints[GraspStatus.grasp_pose]

# # # pre_gripper = grasp_man.get_transformed_gripper_fk(pre_grasp_pose, is_tcp=False)
# # # grasp_man.visualize_gripper(ax, pre_gripper, alpha=0.5, color='red')
# # # grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_grasp_pose), axis=[1,1,1], scale=0.1)
# # gripper = grasp_man.get_transformed_gripper_fk(grasp_pose, is_tcp=False)
# # grasp_man.visualize_gripper(ax, gripper, alpha=0.5, color='blue')
# # grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(grasp_pose), axis=[1,1,1], scale=0.1)

# # support_poses = grasp_man.generate_supports(obj_mesh2, obs_pos2.h_mat, 10, obj_mesh1, obs_pos1.h_mat, 10)
# # release_pose = grasp_man.filter_supports(support_poses)
# # plt.plot_vertices(ax, grasp_man.obj_center_point, s=10)
# # plt.plot_vertices(ax, grasp_man.obj_support_point, s=10)
# # gripper = grasp_man.get_transformed_gripper_fk(release_pose, is_tcp=False)
# # grasp_man.visualize_gripper(ax, gripper, alpha=0.5, color='blue')

# # plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=grasp_man.obj_post_release_pose, alpha=0.2, color='orange')

# # plt.show_figure()
# # ########
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=obs_pos1.h_mat, alpha=0.5, color='orange')
plt.plot_mesh(ax=ax, mesh=obj_mesh2, h_mat=obs_pos2.h_mat, alpha=0.2)
plt.plot_mesh(ax=ax, mesh=obj_mesh3, h_mat=obs_pos3.h_mat, alpha=0.2)

for i, (name, info) in enumerate(objects.grasp_objects.items()):

    grasp_object_info = objects.get_info(name)
    support_object_info = objects.get_info(list(objects.support_objects.keys())[0])
    waypoints = grasp_man.get_grasp_waypoints(grasp_object_info, limit_angle=0.05, num_grasp=10, n_trials=10)
    pre_grasp_pose = waypoints[GraspStatus.pre_grasp_pose]
    grasp_pose = waypoints[GraspStatus.grasp_pose]

    gripper = grasp_man.get_transformed_gripper_fk(pre_grasp_pose, is_tcp=False)
    grasp_man.visualize_gripper(ax, gripper, alpha=0.5, color='blue')
    grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_grasp_pose), axis=[1,1,1], scale=0.1)
    gripper = grasp_man.get_transformed_gripper_fk(grasp_pose, is_tcp=False)

    grasp_man.visualize_gripper(ax, gripper, alpha=0.5, color='blue')
    grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(grasp_pose), axis=[1,1,1], scale=0.1)

    waypoints = grasp_man.get_release_waypoints(support_object_info, 10, grasp_object_info, 10, n_trials=10)
    pre_release_pose = waypoints[GraspStatus.pre_release_pose]
    release_pose = waypoints[GraspStatus.release_pose]

    plt.plot_vertices(ax, grasp_man.obj_center_point)
    plt.plot_vertices(ax, grasp_man.obj_support_point)
    gripper = grasp_man.get_transformed_gripper_fk(release_pose, is_tcp=False)

    grasp_man.visualize_gripper(ax, gripper, alpha=0.5, color='red')
    grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(release_pose), axis=[1,1,1], scale=0.1)

    gripper = grasp_man.get_transformed_gripper_fk(grasp_man.pre_release_pose, is_tcp=False)
    grasp_man.visualize_gripper(ax, gripper, alpha=0.5, color='red')
    grasp_man.visualize_axis(ax, grasp_man.get_tcp_h_mat_from_eef(pre_release_pose), axis=[1,1,1], scale=0.1)

    plt.plot_mesh(ax=ax, mesh=obj_mesh1, h_mat=grasp_man.obj_post_release_pose, alpha=0.2, color='blue')
    plt.show_figure()