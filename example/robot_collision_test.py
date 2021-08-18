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

robot.set_desired_tree("base", "left_wrist")
head_thetas = np.zeros(1)
if robot.joint_safety:
    left_arm_thetas = np.clip(np.random.randn(
        7), robot.joint_limits_lower[:7], robot.joint_limits_upper[:7])

# left_arm_thetas = np.array([-np.pi/4, 0, 0, np.pi, 0, 0, 0])
left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
left_arm_fk = robot.forward_kinematics(left_arm_thetas)
target_pos = left_arm_fk["left_wrist"].matrix()

_, ax = plt.init_3d_figure("Target")
plt.plot_robot(robot, left_arm_fk, ax, "Left arm", visible_collision=True,
               visible_mesh=True, mesh_path='../asset/urdf/baxter/')

# plt.plot_collision(robot, left_arm_fk, ax)
ax.legend()
plt.show_figure()

