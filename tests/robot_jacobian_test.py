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


file_path = '../asset/urdf/baxter/baxter.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# baxter_example
head_thetas = [0.0]
right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

robot.set_desired_tree("base", "left_wrist")

fk = robot.forward_kinematics(left_arm_thetas)
J = robot.jacobian(fk, left_arm_thetas)
print(J)

robot.set_desired_tree("base", "right_wrist")
fk = robot.forward_kinematics(right_arm_thetas)
J = robot.jacobian(fk, right_arm_thetas)
print(J)
