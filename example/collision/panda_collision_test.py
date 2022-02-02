import numpy as np
import trimesh
import yaml
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene


file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))

custom_fpath = '../../asset/config/panda_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = np.array([0, np.pi/16.0, 0.00, -np.pi/2.0 - np.pi/3.0, 0.00, np.pi-0.2, -np.pi/4])
init_qpos = np.array([8.41803072e-02, -1.57518581e-07, -8.41802951e-02, -1.57080031e+00,
 -2.66881047e-08,  1.86750033e+00,  2.02461868e-08])
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, fk, geom="visual")
print(c_manager._filter_names)
c_manager.show_collision_info()
goal_qpos = np.array([ 0.00872548,  0.12562256, -0.81809503, -1.53245947,  2.48667667,  2.6287517, -1.93698104])
goal_fk = robot.forward_kin(goal_qpos)

for link, transform in goal_fk.items():
    if link in c_manager._objs:
        transform = transform.h_mat
        if c_manager.geom == "visual":
            h_mat = np.dot(transform, robot.links[link].visual.offset.h_mat)
        else:
            h_mat = np.dot(transform, robot.links[link].collision.offset.h_mat)
        c_manager.set_transform(name=link, h_mat=h_mat)

result, name = c_manager.in_collision_internal(return_names=True, return_data=False)
print(result, name)

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
test_mesh = trimesh.load_mesh(milk_path)

o_manager = CollisionManager(milk_path)
o_manager.add_object("milk1", gtype="mesh", gparam=test_mesh, h_mat=Transform(pos=[0.1, 0, 0.4]).h_mat)
o_manager.add_object("milk2", gtype="mesh", gparam=test_mesh, h_mat=Transform(pos=[0.4, 0, 0.4]).h_mat)

scene = trimesh.Scene()
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)
scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

scene.add_geometry(test_mesh, node_name="milk1", transform=Transform(pos=[0.1, 0, 0.4]).h_mat)
scene.add_geometry(test_mesh, node_name="milk2", transform=Transform(pos=[0.4, 0, 0.4]).h_mat)

scene.show()
