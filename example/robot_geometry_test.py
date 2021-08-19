import sys
import os
import numpy as np
# pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
# sys.path.append(pykin_path)
file_path = '../asset/urdf/baxter/baxter.urdf'
from pykin import robot
from pykin.robot import Robot
from pykin.kinematics.transform import Transform
from pykin.utils import plot as plt

robot = Robot(file_path, Transform(
    rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=False)

head_thetas = np.zeros(1)
# right_arm_thetas = np.array([0, np.pi, 0, 0, 0, 0, 0])
right_arm_thetas = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0, 0])
left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
fk = robot.forward_kinematics(thetas)

robot.set_geomtry(fk=fk, visible=False)
plt.show_figure()
