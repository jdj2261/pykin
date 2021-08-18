import sys
import os
import numpy as np
from pprint import pprint
#pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
#sys.path.append(pykin_path)
from pykin.utils import plot as plt
from pykin.kinematics import transform as tf
from pykin.robot import Robot
from pykin import robot

file_path = '../asset/urdf/sawyer/sawyer.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=False)
print(robot.get_active_joint_names)
print(robot)
# robot.set_desired_tree("base", "right_wrist")

# sawyer_example
target_thetas = [np.pi/3, 0, 0, 0, 0, 0, 0, 0]
init_thetas = np.random.randn(8)
fk = robot.forward_kinematics(target_thetas)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot, fk, ax, "sawyer", visible_mesh=True, mesh_path='../asset/urdf/sawyer/')
ax.legend()
plt.show_figure()
