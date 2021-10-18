import sys
import math
import numpy as np
import random

import os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.robots.single_arm import SingleArm
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.kinematics.transform import Transform
from pykin.utils.fcl_utils import FclManager
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import get_robot_geom, limit_joints
from pykin.utils.transform_utils import get_homogeneous_matrix


file_path = '../../asset/urdf/iiwa14/iiwa14.urdf'
if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'

fig, ax = plt.init_3d_figure("URDF")
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name("iiwa_link_0", "iiwa_link_ee")

# set target joints angle
target_thetas = np.array([np.random.uniform(-np.pi, np.pi) for _ in range(robot.dof)])
init_thetas = np.random.randn(robot.dof)

target_transformations = robot.forward_kin(target_thetas)

init_q_space = np.zeros(robot.dof)
target_pose = robot.eef_pose

target_q_space = robot.inverse_kin(
    init_thetas, 
    target_pose, 
    method="LM", 
    maxIter=100)

print(init_q_space, target_q_space)

print(robot.joint_limits_lower)

planner = RRTStarPlanner(
    robot=robot,
    obstacles=[],
    max_iter=50,
    fcl_manager=FclManager()
)

path = {}
planner.setup_start_goal_joint(init_q_space, target_q_space)
path = planner.generate_path()
