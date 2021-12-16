import numpy as np
import trimesh
import yaml
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
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=150)

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")
# init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])
# fk = robot.forward_kin(np.array(init_qpos))

# eef_pose = [ 0.7, -0.3, 0.99, -2.50435769e-02,  
#             9.21807347e-01,  3.82091147e-01, -6.04184456e-02]

scale_factor = 1
gm = GraspManager(max_width=0.08 * scale_factor)
mesh = trimesh.load(pykin_path+'/asset/objects/meshes/can.stl')
mesh.apply_scale(scale_factor)
offset_pos = np.array([0.6, -0.3, 0.87])
mesh.apply_translation(offset_pos)

while True:
    vertices, normals = gm.surface_sampling(mesh, n_samples=2)
    if gm.is_force_closure(vertices, normals, limit_angle=0.02):
        break

contact_points = vertices

p1 = contact_points[0]
p2 = contact_points[1]
center_point = (p1 + p2) / 2

distance = np.linalg.norm(p1-p2)

mesh_point, _, _ = trimesh.proximity.closest_point(mesh, [center_point])

y = gm.normalize(p1 - p2)
z = center_point - mesh_point[0] # point into object
z = gm.normalize(z - gm.projection(z, y)) # Gram-Schmidt
x = gm.normalize(np.cross(y, z))

plt.plot_mesh(ax=ax, mesh=mesh, alpha=0.1, color=[0.5, 0, 0])
plt.plot_vertices(ax, contact_points)
plt.plot_vertices(ax, mesh_point)
# plt.plot_normal_vector(ax, contact_points, -normals, scale=0.1)     
plt.plot_normal_vector(ax, mesh_point, x, scale=0.1, edgecolor="red")    
plt.plot_normal_vector(ax, mesh_point, y, scale=0.1, edgecolor="green")    
plt.plot_normal_vector(ax, mesh_point, z, scale=0.1, edgecolor="blue")  

eef_pose = np.eye(4)
eef_pose[:3,0] = x
eef_pose[:3,1] = y
eef_pose[:3,2] = z
eef_pose[:3,3] = mesh_point

t_eef_pose = t_utils.get_pose_from_homogeneous(eef_pose)
init_qpos = robot.inverse_kin(np.random.randn(7), t_eef_pose)

while not robot.check_limit_joint(init_qpos):
    init_qpos = robot.inverse_kin(np.random.randn(len(init_qpos)), t_eef_pose, method="LM")

fk = robot.forward_kin(np.array(init_qpos))

gripper_name = ["right_gripper", "leftfinger", "rightfinger"]
mesh_path = pykin_path+"/asset/urdf/panda/"

for link, transform in fk.items():
    if "pedestal" in link:
        continue
    if robot.links[link].collision.gtype == "mesh":
        mesh_name = mesh_path + robot.links[link].collision.gparam.get('filename')
        mesh = trimesh.load_mesh(mesh_name)
        A2B = np.dot(transform.h_mat, robot.links[link].collision.offset.h_mat)
        color = robot.links[link].collision.gparam.get('color')

        if color is None:
            color = np.array([0.2, 0, 0])
        else:
            color = np.array([color for color in color.values()]).flatten()
            if "link" in link:
                color = np.array([0.2, 0.2, 0.2])
        mesh.visual.face_colors = color
        # if link in gripper_name:
        plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=0.1, color=color)
        # scene.add_geometry(mesh, transform=A2B)  

gripper_pose = fk["panda_right_hand"].h_mat

gripper_pos = gripper_pose[:3, 3]
gripper_ori_x = gripper_pose[:3, 0]
gripper_ori_y = gripper_pose[:3, 1]
gripper_ori_z = gripper_pose[:3, 2]

print(gripper_pose, eef_pose)

plt.plot_vertices(ax, gripper_pos)   
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_x, scale=0.2, edgecolor="red")    
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_y, scale=0.2, edgecolor="green")    
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_z, scale=0.2, edgecolor="blue")   
 
plt.show_figure()