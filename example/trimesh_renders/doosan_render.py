import numpy as np
import trimesh
import os

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene

current_file_path = os.path.abspath(os.path.dirname(__file__))

from pykin.utils import plot_utils as p_utils


file_path = 'urdf/doosan/doosan.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(robot, geom="visual")
c_manager.show_collision_info()

goal_qpos = np.array([ 0,  0, 0, 0,  0,  0])
robot.set_transform(goal_qpos)

for link, info in robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])
        
milk_path = current_file_path + "/../../pykin/asset/objects/meshes/milk.stl"
test_mesh = trimesh.load_mesh(milk_path)

o_manager = CollisionManager()
o_manager.add_object("milk1", gtype="mesh", gparam=test_mesh, h_mat=Transform(pos=[0.1, 0, 0.4]).h_mat)
o_manager.add_object("milk2", gtype="mesh", gparam=test_mesh, h_mat=Transform(pos=[0.4, 0, 0.4]).h_mat)

target_thetas = [0, 0, 0, 0, 0, 0]
robot.set_transform(target_thetas)

scene = trimesh.Scene()
scene = apply_robot_to_scene(trimesh_scene=scene, robot=robot, geom=c_manager.geom)
scene.show()