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
from pykin.utils.obstacle_utils import Obstacle
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene
import pykin.utils.plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
scene = trimesh.Scene()

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")

init_qpos = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk, geom="collision")
c_manager.filter_contact_names(robot, fk)

obs_pos1 = np.array([0.6, 0, 0.78])
obs_pos2 = np.array([0.4, 0.24, 0.0])

obj_mesh1 = trimesh.load(pykin_path+'/asset/objects/meshes/milk.stl')
obj_mesh2 = trimesh.load(pykin_path+'/asset/objects/meshes/custom_table.stl')
obj_mesh2.apply_scale(0.01)

print(obj_mesh1.bounding_box.extents)
print(obj_mesh2.bounding_box.extents)

obs = Obstacle()
o_manager = CollisionManager()
obs(name="can", gtype="mesh", gparam=obj_mesh1, transform=Transform(pos=obs_pos1))
obs(name="table", gtype="mesh", gparam=obj_mesh2, transform=Transform(pos=obs_pos2))
o_manager.add_object("can", gtype="mesh", gparam=obj_mesh1, transform=Transform(pos=obs_pos1).pos)
o_manager.add_object("table", gtype="mesh", gparam=obj_mesh2, transform=Transform(pos=obs_pos2).pos)

g_manager = GraspManager(max_width=0.08, self_c_manager=c_manager, obstacle_c_manager= o_manager)
plt.plot_mesh(ax=ax, mesh=obj_mesh1, A2B=Transform(pos=obs_pos1).h_mat, alpha=0.2)

is_grasp_success = False
while True:
    poses = []
    grasp_poses = g_manager.compute_grasp_pose(
        obj_mesh1, 
        obs_pose=obs_pos1, 
        approach_distance=0.08, 
        limit_angle=0.05, 
        n_trials=5)
    for grasp_pose in grasp_poses:
        transforms, is_grasp_success = g_manager.get_grasp_posture(robot, grasp_pose, n_steps=1, epsilon=0.1)
        pre_transforms, is_post_grasp_success = g_manager.get_pre_grasp_posture(robot, grasp_pose, desired_distance=0.08, n_steps=1, epsilon=0.1)
        poses.append(grasp_pose[:3, 3])
        if is_grasp_success and is_post_grasp_success:
            break
    if is_grasp_success and is_post_grasp_success:
        break

eef_pose = robot.get_eef_pose(transforms)
pre_eef_pose = robot.get_eef_pose(pre_transforms)

print("result!!")
qpos = robot.get_result_qpos(init_qpos, eef_pose)
pre_qpos = robot.get_result_qpos(init_qpos, pre_eef_pose)
transforms = robot.forward_kin(qpos)
pre_transforms = robot.forward_kin(pre_qpos)

plt.plot_vertices(ax, np.array(poses))

gripper_name = ["right_gripper", "leftfinger", "rightfinger"]
mesh_path = pykin_path+"/asset/urdf/panda/"
for link, transform in transforms.items():
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
        plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=0.5, color=color)
        # scene.add_geometry(mesh, transform=A2B)  


for link, transform in pre_transforms.items():
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
        plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=1.0, color=color)

gripper_pose = transforms["panda_right_hand"].h_mat

gripper_pos = gripper_pose[:3, 3]
gripper_ori_x = gripper_pose[:3, 0]
gripper_ori_y = gripper_pose[:3, 1]
gripper_ori_z = gripper_pose[:3, 2]

g_manager.visualize_grasp_pose(ax)
plt.plot_basis(robot, ax)
plt.plot_vertices(ax, gripper_pos)   
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_x, scale=0.2, edgecolor="red")    
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_y, scale=0.2, edgecolor="green")    
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_z, scale=0.2, edgecolor="blue")   

gripper_pose = pre_transforms["panda_right_hand"].h_mat

gripper_pos = gripper_pose[:3, 3]
gripper_ori_x = gripper_pose[:3, 0]
gripper_ori_y = gripper_pose[:3, 1]
gripper_ori_z = gripper_pose[:3, 2]

plt.plot_vertices(ax, gripper_pos)   
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_x, scale=0.2, edgecolor="red")    
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_y, scale=0.2, edgecolor="green")    
plt.plot_normal_vector(ax, gripper_pos, gripper_ori_z, scale=0.2, edgecolor="blue")   


plt.show_figure()