import numpy as np
import sys, os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../../" )
sys.path.append(pykin_path)

import trimesh

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils.collision_utils import CollisionManager
from pykin.utils.mesh_utils import get_mesh_path
from pykin.utils.transform_utils import get_transform_to_visual

file_path = '../../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
# robot.setup_link_name("base", "right_l6")

fk = robot.forward_kin(np.zeros(8))

collision_manager = CollisionManager()
for link, transformation in fk.items():
    if robot.links[link].visual.gtype != "mesh":
        continue

    mesh_name = robot.links[link].visual.gparam.get('filename')
    file_name = get_mesh_path(mesh_name, robot.robot_name)
    mesh = trimesh.load(file_name)
    A2B = get_transform_to_visual(
        transformation.h_mat, robot.visual_offset(link).h_mat)
    collision_manager.add_object(name=robot.links[link].name, gtype="mesh", gparam=mesh, transform=A2B)

    # tri_manager.add_object(name=robot.links[link].name, mesh=mesh, transform=A2B)
    visual_color = robot.links[link].visual.gparam.get('color')
    color = np.array([0.2, 0.2, 0.2, 1.])
    if visual_color is not None:
        color = np.array([color for color in visual_color.values()]).flatten()

result, objs_in_collision, contact_data = collision_manager.collision_check(return_names=True, return_data=True)
