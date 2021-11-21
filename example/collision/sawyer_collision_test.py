import numpy as np
import yaml
import sys, os

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene

import trimesh

custom_fpath = '../../asset/config/sawyer_init_params.yaml'
with open(custom_fpath) as f:
            controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, -0.5]))

# fk = robot.forward_kin(np.array([0, np.pi/2, np.pi/2, np.pi/3, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2]))
fk = robot.forward_kin(np.array(np.concatenate((np.zeros(1), init_qpos))))
# fk = robot.forward_kin(np.zeros(8))
mesh_path = pykin_path+"/asset/urdf/sawyer/"

# init_trainsform
# 로봇의 경우 첫 collision은 제외 일어난 이름은 제외 
c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)
test, name, data = c_manager.in_collision_internal(return_names=True, return_data=True)

scene = trimesh.Scene()
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)
scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

# scene.show()
milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
test_mesh = trimesh.load_mesh(milk_path)
scene.add_geometry(test_mesh, node_name="milk1", transform=Transform(pos=[0.0, 0, 0.1]).h_mat)
scene.add_geometry(test_mesh, node_name="milk2", transform=Transform(pos=[0.1, 0, 0.1]).h_mat)

o_manager = CollisionManager(milk_path)
o_manager.add_object("milk1", gtype="mesh", gparam=test_mesh, transform=Transform(pos=[0.0, 0, 0.1]).h_mat)
o_manager.add_object("milk2", gtype="mesh", gparam=test_mesh, transform=Transform(pos=[0.1, 0, 0.1]).h_mat)
test, name, data = o_manager.in_collision_internal(return_names=True, return_data=True)

result = o_manager.get_distances_other(c_manager)
print(result)
for (a,b), dis in result.items():
    if dis <= 0.0:
        print(a,b, dis)

# # table_path = pykin_path+"/asset/objects/meshes/twin_tower_goal.stl"
# # table_mesh = trimesh.load_mesh(table_path)
# # table_mesh.apply_scale(0.001)
# # scene.add_geometry(table_mesh, transform=Transform(pos=[-1, 0, 0]).h_mat)

scene.show()