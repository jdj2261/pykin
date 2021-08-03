import pykin.robot
from pykin.robot import Robot

file_name = '../../asset/urdf/baxter.urdf'

robot = Robot(file_name)
robot.show_robot_info()
