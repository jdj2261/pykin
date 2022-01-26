import numpy as np
import sys, os
import trimesh

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.tasks.grasp import GraspManager, GraspStatus
from pykin.utils.task_utils import get_relative_transform
from pykin.utils.object_utils import ObjectManager
import pykin.utils.plot_utils as plt

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")

init_qpos = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, fk)

grasp_obj1_pose = Transform(pos=np.array([0.6, 0.2, 0.77]), rot=np.array([0, np.pi/2, np.pi/2]))
grasp_obj2_pose = Transform(pos=np.array([0.4, 0.2, 0.77]), rot=np.array([0, np.pi/2, np.pi/2]))
grasp_obj3_pose = Transform(pos=np.array([0.6, 0.1, 0.77]), rot=np.array([0, np.pi/2, np.pi/2]))

obs_pos2 = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
obs_pos3 = Transform(pos=np.array([0.4, 0.24, 0.0]))

obj_mesh1 = trimesh.load(pykin_path+'/asset/objects/meshes/square_box.stl')
obj_mesh2 = trimesh.load(pykin_path+'/asset/objects/meshes/box_goal.stl')
obj_mesh3 = trimesh.load(pykin_path+'/asset/objects/meshes/custom_table.stl')

obj_mesh1.apply_scale(0.001)
obj_mesh2.apply_scale(0.001)
obj_mesh3.apply_scale(0.01)

objects = ObjectManager()
objects.add_object(name="obj_1", gtype="mesh", gparam=obj_mesh1, transform=grasp_obj1_pose.h_mat, for_grasp=True)
objects.add_object(name="obj_2", gtype="mesh", gparam=obj_mesh1, transform=grasp_obj2_pose.h_mat, for_grasp=True)
# objects.add_object(name="obj_3", gtype="mesh", gparam=obj_mesh1, transform=grasp_obj3_pose.h_mat, for_grasp=True)
objects.add_object(name="box", gtype="mesh", gparam=obj_mesh2, transform=obs_pos2.h_mat, for_support=True)
objects.add_object(name="table", gtype="mesh", gparam=obj_mesh3, transform=obs_pos3.h_mat)

o_manager = CollisionManager()
o_manager.setup_object_collision(objects)

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

####################################################################################
#generate_grasps
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
for i, (name, info) in enumerate(objects.grasp_objects.items()):
    
    if i%3 == 0:
        color ='red'
    elif i%3 == 1:
        color = 'blue'
    else:
        color = 'green'
        
    grasp_object_info = objects.get_info(name)
    support_obj_info = objects.get_info(list(objects.support_objects.keys())[0])
    grasp_waypoints = grasp_man.get_grasp_waypoints(
        obj_info=grasp_object_info, 
        limit_angle=0.05, 
        num_grasp=10, 
        n_trials=10)

    release_waypoints = grasp_man.get_release_waypoints(
        obj_info_on_sup=support_obj_info, 
        n_samples_on_sup=20, 
        obj_info_for_sup=grasp_object_info, 
        n_samples_for_sup=20, 
        n_trials=10)
    
    grasp_pose = grasp_waypoints[GraspStatus.grasp_pose]
    pre_grasp_pose = grasp_waypoints[GraspStatus.pre_grasp_pose]
    
    release_pose = release_waypoints[GraspStatus.release_pose]
    pre_release_pose = release_waypoints[GraspStatus.pre_release_pose]

    T = get_relative_transform(grasp_pose, info[2])
    obj_grasp_pos_transformed = np.dot(pre_grasp_pose, T)
    obj_pre_release_pos_transformed = np.dot(pre_release_pose, T)
    obj_release_pos_transformed = np.dot(release_pose, T)

    grasp_man.visualize_axis(ax, pre_grasp_pose, visible_basis=True)
    grasp_man.visualize_axis(ax, grasp_pose, visible_basis=True)

    grasp_man.visualize_axis(ax, grasp_pose, visible_basis=True)
    gripper = grasp_man.get_gripper_transformed(grasp_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')
    
    grasp_man.visualize_axis(ax, pre_grasp_pose, visible_basis=True)
    gripper = grasp_man.get_gripper_transformed(pre_grasp_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')

    grasp_man.visualize_axis(ax, release_pose, visible_basis=True)
    gripper = grasp_man.get_gripper_transformed(release_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')

    grasp_man.visualize_axis(ax, pre_release_pose, visible_basis=True)
    gripper = grasp_man.get_gripper_transformed(pre_release_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')

    plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obj_grasp_pos_transformed, alpha=0.2, color=color)
    plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obj_release_pos_transformed, alpha=0.2, color=color)
    plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obj_pre_release_pos_transformed, alpha=0.2, color=color)
    plt.plot_mesh(ax=ax, mesh=info[1], A2B=info[2], alpha=0.2, color=color)
    # plt.plot_mesh(ax, mesh=info[1], A2B=grasp_man.result_obj_pose, alpha=0.5, color='red')

plt.plot_mesh(ax=ax, mesh=obj_mesh2, A2B=obs_pos2.h_mat, alpha=0.2)
plt.plot_mesh(ax=ax, mesh=obj_mesh3, A2B=obs_pos3.h_mat, alpha=0.2)
plt.show_figure()
# test = np.eye(4)
# for i, (eef_pose, gripper) in enumerate(grasp_poses):

#     test[:3, 3] = np.array([0, -0.1, 0.1])
#     eef_pose_transfomred = np.dot(eef_pose, test)
#     T = get_relative_transform(eef_pose, obs_pos1.h_mat)
#     obj_pos_transformed = np.dot(eef_pose_transfomred, T)

#     grasp_man.visualize_axis(ax, eef_pose)
#     grasp_man.visualize_axis(ax, eef_pose_transfomred)
#     plt.plot_axis(ax, obs_pos1.h_mat)
#     plt.plot_axis(ax, obj_pos_transformed)
#     plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=obj_pos_transformed, alpha=0.2)
# plt.show_figure()
