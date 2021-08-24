import sys
import os
import numpy as np

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robot import Robot
from pykin.utils import plot_utils as plt

file_path = '../asset/urdf/baxter/baxter.urdf'
# file_path = '../asset/urdf/sawyer.urdf'

robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

head_thetas = np.zeros(1)
right_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
robot_transformations = robot.kin.forward_kinematics(thetas)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot,
               ax, "baxter", 
               visible_visual=False, 
               visible_collision=True,
               mesh_path='../asset/urdf/baxter/')

ax.legend()
plt.show_figure()



