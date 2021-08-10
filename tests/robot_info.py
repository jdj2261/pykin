import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)
import pykin.robot
from pykin.robot import Robot

file_path = '../asset/urdf/baxter/baxter.urdf'

robot = Robot(file_path)
robot.show_robot_info()

# print(robot.joints)
# print(robot.links)
# print(robot.tree.root)
# print(robot.num_links)
# print(robot.num_active_joints)
# print(robot.get_active_joint_names) 
