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

file_path = '../asset/urdf/iiwa14.urdf'

robot = Robot(file_path, tf.Transform(
    rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=True)

# panda_example
target_thetas = [0, np.pi/2, 0, 0, 0, 0, 0]
init_thetas = [0, 0, 0, 0, 0, 0, 0]

print(robot.get_active_joint_names)
robot.set_desired_tree("iiwa_link_0", "iiwa_link_ee")
fk = robot.forward_kinematics(target_thetas)
print(fk)

_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "iiwa14")
ax.legend()
plt.show_figure()

target_pose = np.concatenate(
    (fk["iiwa_link_ee"].pos, fk["iiwa_link_ee"].rot))
ik_result = robot.inverse_kinematics(
    init_thetas, target_pose, method="numerical")
print(ik_result)

robot.desired_frame = None
fk = robot.forward_kinematics(ik_result)
_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "iiwa14")
ax.legend()
plt.show_figure()
