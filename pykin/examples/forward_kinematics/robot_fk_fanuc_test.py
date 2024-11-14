import numpy as np
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils

import numpy as np
import trimesh
import os

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene

current_file_path = os.path.abspath(os.path.dirname(__file__))
file_path = "urdf/fanuc/fanuc_r2000ic_165f.urdf"
goal_qpos = np.array([0, np.pi/2, 0, 0, 0, 0])

# Visual geometry 케이스
robot_visual = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
c_manager_visual = CollisionManager(is_robot=True)
c_manager_visual.setup_robot_collision(robot_visual, geom="visual")
robot_visual.setup_link_name("base_link", "tool0")
robot_visual.set_transform(goal_qpos)
fk_visual = robot_visual.forward_kin(goal_qpos)
print("Visual geometry Tool0 H-matrix:")
print(fk_visual['tool0'].h_mat)
print("\n")

# Collision geometry 케이스
robot_collision = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
c_manager_collision = CollisionManager(is_robot=True)
c_manager_collision.setup_robot_collision(robot_collision, geom="collision")
robot_collision.setup_link_name("base_link", "tool0")
robot_collision.set_transform(goal_qpos)
fk_collision = robot_collision.forward_kin(goal_qpos)
print("Collision geometry Tool0 H-matrix:")
print(fk_collision['tool0'].h_mat)

# 두 행렬의 차이 계산
diff = np.abs(fk_visual['tool0'].h_mat - fk_collision['tool0'].h_mat)
print("\nDifference between matrices:")
print(diff)
print("\nMaximum difference:", np.max(diff))


_, ax = p_utils.init_3d_figure("FK")
p_utils.plot_robot(ax=ax, robot=robot_visual, geom="collision", only_visible_geom=True, alpha=1)
p_utils.show_figure()
