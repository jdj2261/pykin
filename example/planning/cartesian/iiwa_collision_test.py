import numpy as np
import sys, os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils.collision_utils import CollisionManager
from pykin.utils.plot_utils import convert_trimesh_scene
from pykin.utils.kin_utils import get_robot_collision_geom
from pykin.utils import plot_utils as plt
import trimesh
import fcl
file_path = '../../../asset/urdf/iiwa14/iiwa14.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# robot.setup_link_name("base", "right_l6")

mesh_path = pykin_path+"/asset/urdf/iiwa14/"
scene = trimesh.Scene()
tri_manager = trimesh.collision.CollisionManager()

# fk = robot.forward_kin([0.0167305,-0.762614,-0.0207622,-2.34352,-0.0305686,1.53975,0.753872])
fk = robot.forward_kin(np.zeros(7))

collision_manager = CollisionManager()
for link, transformation in fk.items():
    if robot.links[link].visual.gtype == "mesh":
        mesh_name = robot.links[link].visual.gparam.get('filename')
        file_name = mesh_path+mesh_name
        mesh = trimesh.load(file_name)
        A2B = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)
        collision_manager.add_object(name=robot.links[link].name, gtype="mesh", gparam=mesh, transform=A2B)

        tri_manager.add_object(name=robot.links[link].name, mesh=mesh, transform=A2B)
        visual_color = robot.links[link].visual.gparam.get('color')
        color = np.array([0.2, 0.2, 0.2, 1.])
        if visual_color is not None:
            color = np.array([color for color in visual_color.values()]).flatten()
            print(robot.links[link].name, color)
        scene = convert_trimesh_scene(scene, file_name, A2B, color)
        scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

# result, objs_in_collision, contact_data = collision_manager.collision_check(return_names=True, return_data=True)
# result, objs_in_collision, contact_data = tri_manager.in_collision_internal(return_names=True, return_data=True)
# distance, names, data = tri_manager.min_distance_internal(return_names=True, return_data=True)
# ddata = fcl.DistanceData()
# collision_manager._manager.distance(ddata, fcl.defaultDistanceCallback)
# print('Closest distance within manager 1?: {}'.format(ddata.result.min_distance))

# print('')
# print(distance, objs_in_collision, names, data)
# print(result, objs_in_collision, len(contact_data))

scene.show()