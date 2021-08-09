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

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=True)
# baxter_example

# set target joints angle
head_thetas =  np.zeros(1)
# right_arm_thetas = np.array([np.pi/2, 0, 0, 0, 0, 0, 0])
# left_arm_thetas = np.array([-np.pi/2, 0, 0, 0, 0, 0, 0])
right_arm_thetas = np.clip(np.random.randn(
    7), robot.joint_limits_lower[:7], robot.joint_limits_upper[:7])
left_arm_thetas = np.clip(np.random.randn(
    7), robot.joint_limits_lower[:7], robot.joint_limits_upper[:7])

print(f"{scolors.OKBLUE}Target Right arm Angle{scolors.ENDC}: \n{right_arm_thetas}")
print(f"{scolors.OKBLUE}Target Left arm Angle{scolors.ENDC}: \n{left_arm_thetas}")
print()
#################################################################################
#                                Forward Kinematics                             #
#################################################################################
thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))

# Cacluate FK
fk = robot.forward_kinematics(thetas)
target_r_pose = np.concatenate(
    (fk["right_wrist"].pos, fk["right_wrist"].rot))
target_l_pose = np.concatenate(
    (fk["left_wrist"].pos, fk["left_wrist"].rot))

r_pose = fk["right_wrist"].matrix()
l_pose = fk["left_wrist"].matrix()

# show FK graph
_, ax = plt.init_3d_figure("FK Result")
plt.plot_robot(robot, fk, ax, "baxter")
ax.legend()
# plt.show_figure()

#################################################################################
#                                Inverse Kinematics                             #
#################################################################################
# Set desired link (root, end)
robot.set_desired_tree("base", "right_wrist")
right_arm_fk = robot.forward_kinematics(right_arm_thetas)

init_right_thetas = np.clip(np.random.randn(7), robot.joint_limits_lower, robot.joint_limits_upper)
init_left_thetas = np.clip(np.random.randn(7), robot.joint_limits_lower, robot.joint_limits_upper)

target_r_pose = np.concatenate(
    (right_arm_fk["right_wrist"].pos, right_arm_fk["right_wrist"].rot))

# Right's arm IK solution by LM
print("ik_right_LM_result")
ik_right_LM_result = robot.inverse_kinematics(
    init_right_thetas, target_r_pose, method="LM", maxIter=100)

# Right's arm IK solution by NR
print("\nik_right_NR_result")
ik_right_NR_result = robot.inverse_kinematics(
    init_right_thetas, target_r_pose, method="NR")

# Set desired link (root, end)
robot.set_desired_tree("base", "left_wrist")
left_arm_fk = robot.forward_kinematics(left_arm_thetas)

target_l_pose = np.concatenate(
    (left_arm_fk["left_wrist"].pos, left_arm_fk["left_wrist"].rot))

# Left's arm IK solution by LM
print("\nik_left_LM_result")
ik_left_LM_result = robot.inverse_kinematics(
    init_left_thetas, target_l_pose, method="LM", maxIter=100)

# Left's arm IK solution by NR
print("\nik_left_NR_result")
ik_left_NR_result = robot.inverse_kinematics(
    init_left_thetas, target_l_pose, method="NR")

print(f"\n{scolors.HEADER}LM Method: Current Right arm Angles{scolors.ENDC}: \n{ik_right_LM_result}")
print(f"{scolors.HEADER}LM Method: Current Left arm Angles{scolors.ENDC}: \n{ik_left_LM_result}")

print(f"\n{scolors.HEADER}NR Method: Current Right arm Angles{scolors.ENDC}: \n{ik_right_NR_result}")
print(f"{scolors.HEADER}NR Method: Current Left arm Angles{scolors.ENDC}: \n{ik_left_NR_result}")


thetas_LM = np.concatenate((head_thetas, ik_right_LM_result, ik_left_LM_result))
robot.desired_frame = None
result_fk_LM = robot.forward_kinematics(thetas_LM)

goal_r_pose_LM = np.concatenate(
    (result_fk_LM["right_wrist"].pos, result_fk_LM["right_wrist"].rot))
goal_l_pose_LM = np.concatenate(
    (result_fk_LM["left_wrist"].pos, result_fk_LM["left_wrist"].rot))

r_pose = right_arm_fk["right_wrist"].matrix()
l_pose = left_arm_fk["left_wrist"].matrix()

r_pose_new_LM = result_fk_LM["right_wrist"].matrix()
l_pose_new_LM = result_fk_LM["left_wrist"].matrix()

thetas_NR = np.concatenate(
    (head_thetas, ik_right_NR_result, ik_left_NR_result))
robot.desired_frame = None
result_fk_NR = robot.forward_kinematics(thetas_NR)

goal_r_pose_NR = np.concatenate(
    (result_fk_NR["right_wrist"].pos, result_fk_NR["right_wrist"].rot))
goal_l_pose_NR = np.concatenate(
    (result_fk_NR["left_wrist"].pos, result_fk_NR["left_wrist"].rot))

r_pose_new_NR = result_fk_NR["right_wrist"].matrix()
l_pose_new_NR = result_fk_NR["left_wrist"].matrix()


print(f"\n{scolors.OKGREEN}Target Right wrist Pose{scolors.ENDC}: \n{r_pose}")
print(f"{scolors.OKGREEN}LM Method: Current Right wrist Pose{scolors.ENDC}: \n{r_pose_new_LM}")
print(f"{scolors.OKGREEN}NR Method: Current Right wrist Pose{scolors.ENDC}: \n{r_pose_new_NR}")
print(f"\n{scolors.OKCYAN}Target Left wrist Pose{scolors.ENDC}: \n{l_pose}")
print(f"{scolors.OKCYAN}LM Method: Current Left wrist Pose{scolors.ENDC}: \n{l_pose_new_LM}")
print(f"{scolors.OKCYAN}NR Method: Current Left wrist Pose{scolors.ENDC}: \n{l_pose_new_NR}")

right_error = np.linalg.norm(
    np.dot(l_pose_new_LM, np.linalg.inv(r_pose)) - np.mat(np.eye(4)))
left_error = np.linalg.norm(
    np.dot(l_pose_new_LM, np.linalg.inv(l_pose)) - np.mat(np.eye(4)))

print(f"\n{scolors.WARNING}Error{scolors.ENDC}: {right_error}, {left_error}")

_, ax = plt.init_3d_figure("LM IK Result")
plt.plot_robot(robot, result_fk_LM, ax, "baxter")
ax.legend()

_, ax = plt.init_3d_figure("NR IK Result")
plt.plot_robot(robot, result_fk_NR, ax, "baxter")
ax.legend()

plt.show_figure()