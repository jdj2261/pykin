import sys
import os
import numpy as np
from pprint import pprint
# pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
# sys.path.append(pykin_path)
from pykin.utils import plot as plt
from pykin.kinematics import transform as tf
from pykin.robot import Robot
from pykin import robot
from pykin.utils.shell_color import ShellColors as scolors
file_path = '../asset/urdf/baxter/baxter.urdf'
# file_path = '../asset/urdf/sawyer.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=True)


head_thetas = np.zeros(1)
right_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
# robot.set_desired_tree("base", "left_wrist")
left_arm_fk = robot.forward_kinematics(thetas)
target_pos = left_arm_fk["left_wrist"].matrix()

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot, left_arm_fk, ax, "left", visible_collision=False,
               visible_mesh=True, mesh_path='../asset/urdf/baxter/')
# ax.legend()
# plt.show_figure()
# plt.plot_collision(robot, left_arm_fk, ax)


