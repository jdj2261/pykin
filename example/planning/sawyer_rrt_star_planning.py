import sys, os
import numpy as np
from itertools import zip_longest

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.kinematics.transform import Transform
from pykin.utils.obstacle_utils import Obstacle
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("base", "right_l6")

obs = Obstacle()

obs(name="box_2", 
    gtype="box",
    gparam=(0.2, 0.2, 0.2),
    gpose=(0.4, 0.65, 0.3))

# set target joints angle
target_thetas = [0, np.pi/2, 0, 0, np.pi/2, 0, 0, 0]
target_transformations = robot.forward_kin(target_thetas)

init_q_space = np.array([0,0,0,0,0,0,0])
target_pose = robot.compute_eef_pose(target_transformations)

planner = RRTStarPlanner(
    robot=robot,
    obstacles=obs,
    delta_distance=0.1,
    epsilon=0.2, 
    max_iter=1000,
    gamma_RRT_star=10,
)


target_pose = np.array([-1.60300000e-01, 4.81000000e-01, -1.93000000e-01,  2.36109170e-06,
  6.42788730e-01,  7.66043503e-01, -2.81383490e-06])
while True:
    input_pose = np.array(list(map(float, input("xyz 좌표를 입력하세요: ").split())))
    target_pose[0] = input_pose[0]
    target_pose[1] = input_pose[1]
    target_pose[2] = input_pose[2]
    print(target_pose)
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
            fig, ax = plt.init_3d_figure()

            done = False
            trajectory_joints = np.array(path)

            for i, current_joint in enumerate(trajectory_joints):

                transformations = robot.forward_kin(np.concatenate((np.zeros(1), current_joint)))
                trajectories.append(transformations)
                eef_poses.append(transformations[robot.eef_name].pos)
            print(["{0:0.8f}".format(i) for i in eef_poses[-1]])
            plt.plot_animation(
                robot,
                trajectories, 
                fig, 
                ax,
                eef_poses=eef_poses,
                obstacles=obs,
                visible_obstacles=True,
                visible_collision=True, 
                interval=100, 
                repeat=True,
                result=None)
            break
            
