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

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")
# init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])
# fk = robot.forward_kin(np.array(init_qpos))

eef_pose = [ 0.7, -0.3, 0.99, -2.50435769e-02,
            9.21807347e-01,  3.82091147e-01, -6.04184456e-02]
init_qpos = robot.inverse_kin(np.random.randn(7), eef_pose)

fk = robot.forward_kin(np.array(init_qpos))

gripper_name = ["right_gripper", "leftfinger", "rightfinger"]
mesh_path = pykin_path+"/asset/urdf/panda/"

scene = trimesh.Scene()
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=150)
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
        mesh.visual.face_colors = color
        plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=1.0, color=color)
        # scene.add_geometry(mesh, transform=A2B)  

# plt.show_figure()
# mesh_path = pykin_path+"/asset/urdf/panda/"
# c_manager = CollisionManager(mesh_path)
# c_manager.filter_contact_names(robot, fk, geom='collision')
# c_manager = apply_robot_to_collision_manager(c_manager, robot, fk, geom='collision')

# goal_qpos = np.array([ 0.00872548,  0.12562256, -0.81809503, -1.53245947,  2.48667667,  2.6287517, -1.93698104])
# goal_fk = robot.forward_kin(goal_qpos)

# # print(goal_fk["rightfinger"].h_mat)
# # print(goal_fk["leftfinger"].h_mat)

# for link, transform in goal_fk.items():
#     if link in c_manager._objs:
#         transform = transform.h_mat
#         A2B = np.dot(transform, robot.links[link].visual.offset.h_mat)
#         c_manager.set_transform(name=link, transform=A2B)

# fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

gm = GraspManager()
mesh = trimesh.load(pykin_path+'/asset/objects/meshes/can.stl')
# mesh.apply_scale(0.001)

while True:
    vertices, normals = gm.surface_sampling(mesh, n_samples=2)
    if gm.is_force_closure(vertices, normals, limit_radian=0.02):
        break

offset_pos = np.array([0.7, -0.3, 0.87])
contact_points = vertices + np.tile(offset_pos, (2,1))
center_point = ((contact_points[0]+contact_points[1])/2).reshape(1,-1)


plt.plot_mesh(ax=ax, mesh=mesh, A2B=Transform(pos=offset_pos).h_mat, alpha=0.1, color=[0.5, 0, 0])
plt.plot_vertices(ax, contact_points)
plt.plot_vertices(ax, center_point)
# plt.plot_normal_vector(ax, vertices, -normals, scale=0.1)    
plt.show_figure()