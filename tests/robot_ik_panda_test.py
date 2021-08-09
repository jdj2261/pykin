
import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)
from pykin.utils import plot as plt
from pykin.kinematics import transform as tf
from pykin.robot import Robot
from pykin import robot

file_path = '../asset/urdf/panda.urdf'

robot = Robot(file_path, tf.Transform(
    rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=True)
robot.set_desired_tree("panda_link0", "panda_hand")

# panda_example
target_thetas = [np.pi/3, 0, 0, 0, 0, 0, 0]
init_thetas = [0, 0, 0, 0, 0, 0, 0]


fk = robot.forward_kinematics(target_thetas)
print(fk)

_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "panda")
ax.legend()
plt.show_figure()

target_pose = np.concatenate(
    (fk["panda_hand"].pos, fk["panda_hand"].rot))

ik_result = robot.inverse_kinematics(
    init_thetas, target_pose, method="LM")
print(ik_result)

# robot.desired_frame = None
fk = robot.forward_kinematics(ik_result)
_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "panda")
ax.legend()
plt.show_figure()
