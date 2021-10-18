from itertools import zip_longest

import sys, os
import numpy as np

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.utils import plot_utils as plt   
   
file_path = '../../asset/urdf/baxter/baxter.urdf'

robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name("base", "right_wrist")
robot.setup_link_name("base", "left_wrist")

# set target joints angle
head_thetas =  np.zeros(1)
right_arm_thetas = np.array([np.pi/4 , 0, -np.pi/4, 0, 0 , 0 ,0])
left_arm_thetas = np.array([np.pi/4 , 0, -np.pi/4, 0, 0 , 0 ,0])

thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))
target_transformations = robot.forward_kin(thetas)

init_q_space = { "right": np.zeros(7), 
                "left" : np.zeros(7)}

target_pose = { "right": robot.eef_pose["right"], 
                "left" : robot.eef_pose["left"]}

target_q_space = robot.inverse_kin(
    np.random.randn(7), 
    target_pose, 
    method="LM", 
    maxIter=100)

# robot.remove_desired_frames()
planner = RRTStarPlanner(
    robot=robot,
    obstacles=[],
    delta_distance=0.1,
    epsilon=0.2, 
    max_iter=100,
    gamma_RRT_star=1,
)


path = {}

for arm in robot.arms:
    planner.setup_start_goal_joint(init_q_space[arm], target_q_space[arm], arm)
    path[arm] = planner.generate_path()

print(path.values())

trajectory_pos = []
trajectories = []
if any(value is None for value in path.values()):
    print("Not created trajectories..")
else:
    current_q_space = { "right": path["right"][-1], "left" : path["left"][-1]}
    trajectory_joints = list(zip_longest(np.array(path["right"]), np.array(path["left"])))

    for i, (right_joint, left_joint) in enumerate(trajectory_joints):

        if right_joint is None:
            right_joint = last_right_joint
        if left_joint is None:
            left_joint = last_left_joint

        last_right_joint = right_joint
        last_left_joint = left_joint

        current_joint = np.concatenate((head_thetas, right_joint, left_joint)) 
        transformations = robot.forward_kin(current_joint)
        trajectory_pos.append(transformations)

    plt.plot_animation(robot, trajectory_pos, interval=1, repeat=False)