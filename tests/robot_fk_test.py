import sys
import os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)

import numpy as np
import pykin.robot
from pykin.robot import Robot
from pprint import pprint
import pykin.kinematics.transform as tf

file_name = '../asset/urdf/baxter.urdf'

robot = Robot(file_name)
robot.offset = tf.Transform(rot = [1.0, 0.0, 0.0, 0.0], pos=[100, 0, 0])
# baxter example
head_thetas = [0.0]
right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

thetas = head_thetas + right_arm_thetas + left_arm_thetas
fk = robot.forward_kinematics(thetas)

"""
If you want to know transformations of all links,
you don't have to write get_desired_tree and desired_frame.
"""
pprint(fk)
# for link, T in fk.items():
#     print(f"link: {link}, pose:{np.concatenate((T.pos, T.rot))} ")


# """
# If you want to know transformation of desired link,
# you must write get_desried_tree.
# """

# robot.get_desired_tree("base", "left_wrist")
# fk = robot.forward_kinematics(left_arm_thetas)
# pprint(fk)
# for link, T in fk.items():
#     print(f"link: {link}, pose:{np.concatenate((T.pos, T.rot))} ")

# """
# If you want to reknow transformations of all links,
# you must write desired_frame.
# """
# robot.desired_frame = None
# fk = robot.forward_kinematics(thetas)
# pprint(fk)
