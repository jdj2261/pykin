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
file_path = '../asset/urdf/baxter/baxter.urdf'
# file_path = '../asset/urdf/sawyer.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# baxter_example
head_thetas = [0.0]
right_arm_thetas = list(np.random.randn(7))
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

thetas = head_thetas + right_arm_thetas + left_arm_thetas
# thetas = [0, 0, 0, 0, 0, 0, 0, 0]
fk = robot.forward_kinematics(thetas)


_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "baxter", mesh_path='../asset/urdf/baxter/')
# plt.plot_robot(robot, fk, ax, "saywer")
ax.legend()
# plt.plot_cylinder(robot, fk, ax, length=0.3, radius=0.1, alpha=0.2)
plt.show_figure()
