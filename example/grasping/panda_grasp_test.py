import numpy as np
import trimesh
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.grasp_utils import GraspManager
from pykin.utils.collision_utils import apply_robot_to_collision_manager

import pykin.utils.plot_utils as plt
import  pykin.utils.transform_utils as t_utils


scene = trimesh.Scene()
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")
# init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])
# fk = robot.forward_kin(np.array(init_qpos))

# eef_pose = [ 0.7, -0.3, 0.99, -2.50435769e-02,  
#             9.21807347e-01,  3.82091147e-01, -6.04184456e-02]

scale_factor = 1
gm = GraspManager(max_width=0.08 * scale_factor)
obj_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/can.stl')
obj_mesh.apply_scale(scale_factor)
offset_pos = np.array([0.6, 0, 0.87])
obj_mesh.apply_translation(offset_pos)
color = obj_mesh.visual.face_colors[0][:3]/255
plt.plot_mesh(ax=ax, mesh=obj_mesh, alpha=0.2, color=color)

for transform in gm.compute_grasp_pose(obj_mesh):
    # print(transform)
    plt.plot_vertices(ax, gm.contact_points)
    plt.plot_vertices(ax, gm.mesh_point, c='red')
    # plt.plot_normal_vector(ax, transform[:3, 3], transform[:3, 0], scale=0.05, edgecolor="red")    
    plt.plot_normal_vector(ax, transform[:3, 3], transform[:3, 1], scale=0.05, edgecolor="green")    
    plt.plot_normal_vector(ax, transform[:3, 3], transform[:3, 2], scale=0.05, edgecolor="blue")  
    plt.plot_vertices(ax, transform[:3, 3])

plt.show_figure()
# while True:
#     grasp_pose = gm.compute_grasp_pose(obj_mesh, 0.08, 0.02)
#     transforms, is_grasp_success = gm.get_grasp_posture(robot, grasp_pose, epsilon=0.5)
#     post_transforms, is_post_grasp_success = gm.get_pre_grasp_posture(robot, grasp_pose, epsilon=0.5)
#     if is_post_grasp_success:
#         break

# gripper_name = ["right_gripper", "leftfinger", "rightfinger"]
# mesh_path = pykin_path+"/asset/urdf/panda/"

# for link, transform in post_transforms.items():
#     if "pedestal" in link:
#         continue
#     if robot.links[link].collision.gtype == "mesh":
#         mesh_name = mesh_path + robot.links[link].collision.gparam.get('filename')
#         mesh = trimesh.load_mesh(mesh_name)
#         A2B = np.dot(transform.h_mat, robot.links[link].collision.offset.h_mat)
#         color = robot.links[link].collision.gparam.get('color')

#         if color is None:
#             color = np.array([0.2, 0, 0])
#         else:
#             color = np.array([color for color in color.values()]).flatten()
#             if "link" in link:
#                 color = np.array([0.2, 0.2, 0.2])
#         mesh.visual.face_colors = color
#         # if link in gripper_name:
#         # plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=1.0, color=color)
#         # scene.add_geometry(mesh, transform=A2B)  

# gripper_pose = post_transforms["panda_right_hand"].h_mat

# gripper_pos = gripper_pose[:3, 3]
# gripper_ori_x = gripper_pose[:3, 0]
# gripper_ori_y = gripper_pose[:3, 1]
# gripper_ori_z = gripper_pose[:3, 2]

# # print(gripper_pose, t_eef_pose)
# plt.plot_basis(robot, ax)
# gm.visualize_grasp_pose(ax)

# plt.plot_vertices(ax, gripper_pos)   
# plt.plot_normal_vector(ax, gripper_pos, gripper_ori_x, scale=0.2, edgecolor="red")    
# plt.plot_normal_vector(ax, gripper_pos, gripper_ori_y, scale=0.2, edgecolor="green")    
# plt.plot_normal_vector(ax, gripper_pos, gripper_ori_z, scale=0.2, edgecolor="blue")   
 
# plt.show_figure()