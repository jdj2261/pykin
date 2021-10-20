import sys, os
import numpy as np
from itertools import zip_longest

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.kinematics.transform import Transform

from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("base", "right_l6")

# set target joints angle
target_thetas = [0, np.pi/2, 0, 0, 0, 0, 0, 0]
target_transformations = robot.forward_kin(target_thetas)

init_q_space = np.array([0,0,0,0,0,0,0])
target_pose = robot.compute_eef_pose(target_transformations)

fig, ax = plt.init_3d_figure()

planner = RRTStarPlanner(
    robot=robot,
    obstacles=[],
    delta_distance=0.1,
    epsilon=0.2, 
    max_iter=300,
    gamma_RRT_star=10,
)

cnt = 0
done = True

while cnt <= 20 and done:

    target_q_space = robot.inverse_kin(
        np.random.randn(7), 
        target_pose, 
        method="LM", 
        maxIter=100)

    path = {}
    planner.setup_start_goal_joint(init_q_space, target_q_space)
    path = planner.generate_path()

    trajectories = []
    eef_poses = []
    if path is None:
        done = True
        cnt += 1
        print(f"{cnt}th try to find path..")
    else:
        done = False
        trajectory_joints = np.array(path)

        for i, current_joint in enumerate(trajectory_joints):

            transformations = robot.forward_kin(np.concatenate((np.zeros(1), current_joint)))
            trajectories.append(transformations)
            eef_poses.append(transformations[robot.eef_name].pos)

        plt.plot_animation(
            robot,
            trajectories, 
            eef_poses,
            fig, 
            ax,
            obstacles=[],
            visible_obstacles=True,
            visible_collision=False, 
            interval=100, 
            repeat=False,
            result=None)
        
