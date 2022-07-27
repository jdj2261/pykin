import numpy as np
import trimesh
import os

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene

current_file_path = os.path.abspath(os.path.dirname(__file__))

file_path = 'urdf/doosan/doosan_with_robotiq140.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
print(robot.robot_name)
c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(robot, geom="visual")
c_manager.show_collision_info()
robot.setup_link_name("base_0", "right_hand")

# goal_qpos = np.array([ 0, 0, np.pi/1.5, 0, np.pi/3,  np.pi/2])
goal_qpos = np.array([ 0, 0, 0, 0, 0,  0])
robot.set_transform(goal_qpos)

scene = trimesh.Scene()
scene = apply_robot_to_scene(trimesh_scene=scene, robot=robot, geom=c_manager.geom)

# robot2 = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
# print(robot.robot_name)
# c_manager = CollisionManager(is_robot=True)
# c_manager.setup_robot_collision(robot2, geom="visual")
# c_manager.show_collision_info()
# robot2.setup_link_name("base_0", "right_hand")

# # goal_qpos = np.array([ 0, 0, np.pi/1.5, 0, np.pi/3,  np.pi/2])
# goal_qpos = np.array([ 0, 0, 0, 0, 0,  0])
# robot2.set_transform(goal_qpos)

# # scene = trimesh.Scene()
# scene = apply_robot_to_scene(trimesh_scene=scene, robot=robot2, geom=c_manager.geom)
# # scene.show()

scene.show()