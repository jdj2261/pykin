import numpy as np
import trimesh
import yaml
import sys, os

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
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
init_qpos =  np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot, fk)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)

goal_qpos = np.array([ 0.00872548,  0.12562256, -0.81809503, -1.53245947,  2.48667667,  2.6287517, -1.93698104])
goal_fk = robot.forward_kin(goal_qpos)

for link, transform in goal_fk.items():
    if link in c_manager._objs:
        transform = transform.h_mat
        A2B = np.dot(transform, robot.links[link].visual.offset.h_mat)
        print(link, A2B)
        c_manager.set_transform(name=link, transform=A2B)

result, objs_in_collision, contact_data = c_manager.in_collision_internal(return_names=True, return_data=True)
distance = c_manager.get_distances_internal()
print(distance)
print(result, objs_in_collision, len(contact_data))

scene = trimesh.Scene()
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)
scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
test_mesh = trimesh.load_mesh(milk_path)
scene.add_geometry(test_mesh, transform=Transform(pos=[0, 0, 0]).h_mat)

table_path = pykin_path+"/asset/objects/meshes/custom_table.stl"
table_mesh = trimesh.load_mesh(table_path)
table_mesh.apply_scale(0.01)
scene.add_geometry(table_mesh, transform=Transform(pos=[0.7, 0, 0]).h_mat)

scene.show()