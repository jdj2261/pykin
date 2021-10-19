from itertools import zip_longest

import sys, os
import numpy as np

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.utils import plot_utils as plt   


fig, ax = plt.init_3d_figure(figsize=(18,9), dpi= 100)

file_path = '../../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name("base", "right_wrist")
robot.setup_link_name("base", "left_wrist")

spheres = {}
radius = 0.1
obstacle_name = "obstacle_sphere_1"
spheres[obstacle_name] = ((0.3, -0.65, 0.3, radius))

obstacle_name = "obstacle_sphere_2"
spheres[obstacle_name] = ((0.4, 0.8, 0.3, 0.08))

planner = RRTStarPlanner(
    robot=robot,
    obstacles=spheres,
    delta_distance=0.5,
    epsilon=0.1, 
    max_iter=600,
    gamma_RRT_star=10,
)

head_thetas =  np.zeros(1)

current_right_joints = np.array([-np.pi/4, 0, 0, 0, 0, 0, 0])
current_left_joints = np.zeros(7)

target_right_joints = np.array([np.pi/3, np.pi/5, np.pi/2, np.pi/7, 0, 0 ,0])
target_left_joints = np.array([np.pi/4 , 0, 0, 0, 0 , 0 ,0])

currrent_joints = np.concatenate((head_thetas, current_right_joints, current_left_joints))
target_joints = np.concatenate((head_thetas ,target_right_joints ,target_left_joints))

init_transformations = robot.forward_kin(currrent_joints)
target_transformations = robot.forward_kin(target_joints)

init_q_space = { "right": current_right_joints, 
                 "left" : current_left_joints}

cnt = 0
done = np.array([False, False])
result = {"right":None, "left":None}

while cnt <= 20 and not done.all():

    cnt += 1
    path = {}
    for i, arm in enumerate(robot.arms):
        if not done[i]:
            target_pose = { arm: robot.compute_eef_pose(target_transformations)[arm]}
            target_q_space = robot.inverse_kin(
            np.random.randn(7), 
            target_pose, 
            method="LM", 
            maxIter=100
            )

            print(f"{cnt} {arm} try to get path")
            
            planner.setup_start_goal_joint(init_q_space[arm], target_q_space[arm], arm, init_transformations)
            path[arm] = planner.generate_path()
            if path[arm] is not None:
                done[i] = True
                result[arm] = path[arm]
                print(f"{arm} path success")

    trajectories = []
    if all(value is not None for value in result.values()):
        done[:] = True
        trajectory_joints = list(zip_longest(np.array(result["right"]), np.array(result["left"])))

        for i, (right_joint, left_joint) in enumerate(trajectory_joints):

            if right_joint is None:
                right_joint = last_right_joint
            if left_joint is None:
                left_joint = last_left_joint

            last_right_joint = right_joint
            last_left_joint = left_joint

            current_joint = np.concatenate((head_thetas, right_joint, left_joint)) 
            transformations = robot.forward_kin(current_joint)
            trajectories.append(transformations)

        plt.plot_animation(
            robot,
            trajectories, 
            fig, 
            ax,
            obstacels=spheres,
            visible_obstacles=True,
            visible_collision=True, 
            interval=100, 
            repeat=False,
            result=None)