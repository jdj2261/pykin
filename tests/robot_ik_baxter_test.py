import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)
from pykin import robot
from pykin.robot import Robot
from pykin.kinematics import transform as tf
from pykin.utils import plot as plt
from pykin.utils.shell_color import ShellColors as scolors
file_path = '../asset/urdf/baxter.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# baxter_example

# set target joints angle
head_thetas =  np.zeros(1)
# right_arm_thetas = [np.pi/6, 0, np.pi/6, 0, 0, 0, 0]
# left_arm_thetas = [-np.pi/6, 0, 0, np.pi/6, 0, -np.pi/6, np.pi/6]
right_arm_thetas = np.random.randn(7)
left_arm_thetas = np.random.randn(7)
print(f"{scolors.OKBLUE}Target Right arm Angle{scolors.ENDC}: \n{right_arm_thetas}")
print(f"{scolors.OKBLUE}Target Left arm Angle{scolors.ENDC}: \n{left_arm_thetas}")

#################################################################################
#                                Forward Kinematics                             #
#################################################################################
# First, show FK graph
thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))
fk = robot.forward_kinematics(thetas)
target_r_pose = np.concatenate(
    (fk["right_wrist"].pos, fk["right_wrist"].rot))
target_l_pose = np.concatenate(
    (fk["left_wrist"].pos, fk["left_wrist"].rot))


r_pose = fk["right_wrist"].matrix()
l_pose = fk["left_wrist"].matrix()
_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "baxter")
ax.legend()
plt.show_figure()

#################################################################################
#                                Inverse Kinematics                             #
#################################################################################
# init joints
init_right_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
init_left_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

robot.set_desired_tree("base", "right_wrist")
right_arm_fk = robot.forward_kinematics(right_arm_thetas)
target_r_pose = np.concatenate(
    (right_arm_fk["right_wrist"].pos, right_arm_fk["right_wrist"].rot))

ik_right_result = robot.inverse_kinematics(
    init_right_thetas, target_r_pose, method="numerical")

robot.set_desired_tree("base", "left_wrist")
left_arm_fk = robot.forward_kinematics(left_arm_thetas)

target_l_pose = np.concatenate(
    (left_arm_fk["left_wrist"].pos, left_arm_fk["left_wrist"].rot))

r_pose = right_arm_fk["right_wrist"].matrix()
l_pose = left_arm_fk["left_wrist"].matrix()

ik_left_result = robot.inverse_kinematics(init_left_thetas, target_l_pose, method="numerical")

print(f"\n{scolors.HEADER}Current Right arm Angles{scolors.ENDC}: \n{ik_right_result}")
print(f"{scolors.HEADER}Current Left arm Angles{scolors.ENDC}: \n{ik_left_result}")

thetas = np.concatenate((head_thetas, ik_right_result, ik_left_result))
robot.desired_frame = None
result_fk = robot.forward_kinematics(thetas)

goal_r_pose = np.concatenate(
    (result_fk["right_wrist"].pos, result_fk["right_wrist"].rot))
goal_l_pose = np.concatenate(
    (result_fk["left_wrist"].pos, result_fk["left_wrist"].rot))


r_pose_new = result_fk["right_wrist"].matrix()
l_pose_new = result_fk["left_wrist"].matrix()

print(f"\n{scolors.OKGREEN}Target Right wrist Pose{scolors.ENDC}: \n{r_pose}")
print(f"{scolors.OKGREEN}Current Right wrist Pose{scolors.ENDC}: \n{r_pose_new}")
print(f"\n{scolors.OKCYAN}Target Light wrist Pose{scolors.ENDC}: \n{l_pose}")
print(f"{scolors.OKCYAN}Current Light wrist Pose{scolors.ENDC}: \n{l_pose_new}")

_, ax = plt.init_3d_figure()
plt.plot_robot(robot, result_fk, ax, "baxter")
ax.legend()
plt.show_figure()

right_error = np.linalg.norm(np.dot(r_pose_new , np.linalg.inv(r_pose)) - np.mat(np.eye(4)))
left_error = np.linalg.norm(np.dot(l_pose_new , np.linalg.inv(l_pose)) - np.mat(np.eye(4)))

print(f"\n{scolors.WARNING}Error{scolors.ENDC}: {right_error}, {left_error}")

