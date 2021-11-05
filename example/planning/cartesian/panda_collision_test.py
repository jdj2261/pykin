import numpy as np
import sys, os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene

import trimesh

file_path = '../../../asset/urdf/panda/panda.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, -1]))

mesh_path = pykin_path+"/asset/urdf/panda/"

collision_manager = CollisionManager(mesh_path)
c_manager = apply_robot_to_collision_manager(collision_manager, robot)

result, objs_in_collision, contact_data = c_manager.collision_check(return_names=True, return_data=True)
print(result, objs_in_collision, len(contact_data))

scene = trimesh.Scene()
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot)
scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
test_mesh = trimesh.load_mesh(milk_path)
scene.add_geometry(test_mesh, transform=Transform(pos=[1, 0, 0]).h_mat)

table_path = pykin_path+"/asset/objects/meshes/custom_table.stl"
table_mesh = trimesh.load_mesh(table_path)
table_mesh.apply_scale(0.01)
scene.add_geometry(table_mesh, transform=Transform(pos=[-1, 0, 0]).h_mat)

scene.show()