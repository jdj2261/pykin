import sys
import os
import numpy as np
from pprint import pprint
# pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
# sys.path.append(pykin_path)
from pykin import robot
from pykin.robot import Robot
from pykin.kinematics import transform as tf
from pykin.utils import plot as plt

file_path = '../asset/urdf/baxter.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# baxter_example
head_thetas = [0.0]
right_arm_thetas = [-np.pi/4, 0, 0, 0, 0, 0, 0]
left_arm_thetas = [np.pi/4, 0, 0, 0, 0, 0, 0]

init_right_thetas = [0, 0, 0, 0, 0, 0, 0]
init_left_thetas = [0, 0, 0, 0, 0, 0, 0]


robot.set_desired_tree("base", "right_wrist")

right_arm_fk = robot.forward_kinematics(right_arm_thetas)

target_r_pose = np.concatenate(
    (right_arm_fk["right_wrist"].pos, right_arm_fk["right_wrist"].rot))

ik_right_result = robot.inverse_kinematics(
    init_right_thetas, target_r_pose, method="numerical")
print(ik_right_result)

robot.set_desired_tree("base", "left_wrist")
left_arm_fk = robot.forward_kinematics(left_arm_thetas)

target_l_pose = np.concatenate(
    (left_arm_fk["left_wrist"].pos, left_arm_fk["left_wrist"].rot))

ik_left_result = robot.inverse_kinematics(init_left_thetas, target_l_pose, method="numerical")
print(ik_left_result)

robot.desired_frame = None
thetas = head_thetas + right_arm_thetas + left_arm_thetas
fk = robot.forward_kinematics(thetas)

_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "baxter")
ax.legend()
plt.show_figure()


thetas = head_thetas + ik_right_result + ik_left_result
fk = robot.forward_kinematics(thetas)

_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "baxter")
ax.legend()
plt.show_figure()
