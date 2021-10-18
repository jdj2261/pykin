import sys, os
import numpy as np
from itertools import zip_longest

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.kinematics.transform import Transform

from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/iiwa14/iiwa14.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name("iiwa_link_0", "iiwa_link_ee")


# set target joints angle
target_thetas = np.array([np.pi/4 , -np.pi/2, -np.pi/2, -np.pi/4, -np.pi/4 , -np.pi/4 ,-np.pi/4])
init_thetas = np.random.randn(robot.dof)
robot.forward_kin(target_thetas)

init_q_space = np.zeros(robot.dof)
target_pose = robot.eef_pose

target_q_space = robot.inverse_kin(
    init_thetas, 
    target_pose, 
    method="LM", 
    maxIter=100)

planner = RRTStarPlanner(
    robot=robot,
    obstacles=[],
    delta_distance=0.1,
    epsilon=0.2, 
    max_iter=500,
    gamma_RRT_star=1,
)

cnt = 0
done = True
while cnt <= 20 and done:
    path = {}
    planner.setup_start_goal_joint(init_q_space, target_q_space)
    path = planner.generate_path()

    trajectories = []
    trajectory_pos = []
    if path is None:
        done = True
        cnt += 1
        print(f"{cnt}th try to find path..")
    else:
        done = False
        trajectory_joints = np.array(path)

        for i, current_joint in enumerate(trajectory_joints):

            transformations = robot.forward_kin(current_joint)
            trajectory_pos.append(transformations)

        plt.plot_animation(robot, trajectory_pos, interval=1, repeat=False)