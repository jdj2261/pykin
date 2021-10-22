from itertools import zip_longest

import sys, os
import numpy as np

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.utils import plot_utils as plt   
from pykin.utils.obstacle_utils import Obstacle
from pykin.utils.kin_utils import ShellColors as scolors

fig, ax = plt.init_3d_figure(figsize=(18,9), dpi= 100)

file_path = '../../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name("base", "right_wrist")
robot.setup_link_name("base", "left_wrist")

obs = Obstacle()

# obs(name="sphere_1", 
#     gtype="sphere",
#     gparam=0.1,
#     gpose=(0.3, -0.65, 0.3))

obs(name="box_1", 
    gtype="box",
    gparam=(0.1, 0.1, 0.3),
    gpose=(0.5, -0.65, 0.3))

obs(name="box_2", 
    gtype="box",
    gparam=(0.1, 0.1, 0.1),
    gpose=(0.4, 0.65, 0.3))

planner = RRTStarPlanner(
    robot=robot,
    obstacles=obs,
    delta_distance=0.1,
    epsilon=0.2, 
    max_iter=500,
    gamma_RRT_star=1.0,
)

head_thetas =  np.zeros(1)

current_right_joints = np.array([-np.pi/4, 0, 0, 0, 0, 0, 0])
current_left_joints = np.array([-np.pi/4, 0, 0, 0, 0, 0, 0])

target_right_joints = np.array([np.pi/2, -np.pi/2, -np.pi/2, np.pi/7, 0, np.pi/7 ,0])
target_left_joints = np.array([np.pi/2 , 0, 0, 0, 0 , 0 ,0])

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
    for i, arm in enumerate(robot.arm_type):
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
                print(f"{scolors.OKGREEN}{arm} path was created successfully\n{scolors.ENDC}")

    trajectories = []
    eef_poses = []
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

            for arm in robot.arm_type:
                eef_poses.append(transformations[robot.eef_name[arm]].pos)

        plt.plot_animation(
            robot,
            trajectories,
            fig=fig, 
            ax=ax,
            eef_poses=eef_poses,
            obstacles=obs,
            visible_obstacles=True,
            visible_collision=False, 
            visible_text=False,
            visible_scatter=False,
            interval=1, 
            repeat=True,
            result=None)