from itertools import zip_longest
import numpy as np

from pykin.robot import Robot
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/baxter/baxter.urdf'
robot = Robot(file_path)

head_thetas = [0]
left_thetas = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0, 0])
right_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

init_thetas = np.random.randn(7)
baxter_thetas = np.concatenate((head_thetas, left_thetas, right_thetas))
transformations = robot.kin.forward_kinematics(baxter_thetas)

target_l_pose = np.hstack((transformations["left_wrist"].pos, transformations["left_wrist"].rot))
target_r_pose = np.hstack((transformations["right_wrist"].pos, transformations["right_wrist"].rot))

robot.set_desired_frame(base_name="base", eef_name="left_wrist")
left_arm_thetas = [0, np.pi/2, 0, -np.pi/2, 0, 0, 0]
init_left_thetas = np.random.randn(7)
left_transformations = robot.kin.forward_kinematics(left_arm_thetas)


target_l_pose = np.concatenate((left_transformations["left_wrist"].pos, left_transformations["left_wrist"].rot))
ik_left_result, trajectory_joints_l = robot.kin.inverse_kinematics(init_left_thetas, 
                                                                    target_l_pose, 
                                                                    method="LM", 
                                                                    maxIter=50)

robot.set_desired_frame(base_name="base", eef_name="right_wrist")
right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
init_right_thetas = np.random.randn(7)
right_transformations = robot.kin.forward_kinematics(right_arm_thetas)
target_r_pose = np.concatenate((right_transformations["right_wrist"].pos, right_transformations["right_wrist"].rot))
ik_right_result, trajectory_joints_r = robot.kin.inverse_kinematics(init_right_thetas, 
                                                                    target_r_pose, 
                                                                    method="LM", 
                                                                    maxIter=50)

trajectory_joints = list(zip_longest(trajectory_joints_l, trajectory_joints_r))

trajectory_pos = []
results = []
robot.reset_desired_frames()
print(trajectory_joints[0])
for i, (left_joint, right_joint) in enumerate(trajectory_joints):

    if left_joint is None:
        left_joint = last_left_joint
    if right_joint is None:
        right_joint = last_right_joint
    last_left_joint = left_joint
    last_right_joint = right_joint

    current_joint = np.concatenate((head_thetas, left_joint, right_joint))
    transformations = robot.kin.forward_kinematics(current_joint)
    trajectory_pos.append(transformations)

plt.plot_animation(robot, trajectory_pos, interval=100, repeat=False)
