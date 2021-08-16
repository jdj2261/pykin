import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)
file_path = '../asset/urdf/baxter/baxter.urdf'
from pykin.utils.shell_color import ShellColors as scolors
from pykin import robot
from pykin.robot import Robot
from pykin.kinematics import transform as tf
from pykin.utils import plot as plt

robot = Robot(file_path, tf.Transform(
    rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=False)

head_thetas = np.zeros(1)
right_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
robot.set_desired_tree("base", "right_wrist")
right_arm_fk = robot.forward_kinematics(right_arm_thetas)
# fk = robot.forward_kinematics(thetas)


# print(right_arm_fk)
_, ax = plt.init_3d_figure("Target")
robot.plot_geomtry(ax, fk=right_arm_fk)
print(robot.geo)

for idx, obj in enumerate(robot.geo.objects):
    print(idx, obj)

ax.legend()
plt.show_figure()
