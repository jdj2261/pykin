import numpy as np
from itertools import zip_longest

from pykin.robots.single_arm import SingleArm
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.kinematics.transform import Transform

from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/panda/panda.urdf'

fig, ax = plt.init_3d_figure()
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name("panda_link0", "panda_link7")

# set target joints angle
target_thetas = np.array([0, np.pi/5, 0, 0, 0, 0, 0])
target_transformations = robot.forward_kin(target_thetas)

init_q_space = np.array([0,0,0,0,0,0,0])
target_pose = robot.compute_eef_pose(target_transformations)

planner = RRTStarPlanner(
    robot=robot,
    obstacles=[],
    delta_distance=0.1,
    epsilon=0.2, 
    max_iter=100,
    gamma_RRT_star=1,
)

cnt = 0
done = True
while done:
    target_q_space = robot.inverse_kin(
        np.random.randn(7), 
        target_pose, 
        method="LM", 
        maxIter=100)

    path = {}
    planner.setup_start_goal_joint(init_q_space, target_q_space)
    path = planner.generate_path()

    result = []
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

            transformations = robot.forward_kin(current_joint)
            trajectories.append(transformations)
            eef_poses.append(transformations[robot.eef_name].pos)

        plt.plot_animation(
            robot,
            trajectories, 
            fig, 
            ax,
            eef_poses=eef_poses,
            obstacles=[],
            visible_obstacles=True,
            visible_collision=True, 
            interval=1, 
            repeat=True,
            result=None)
