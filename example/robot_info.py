import sys
from pykin.robot import Robot

file_path = '../asset/urdf/baxter/baxter.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
robot = Robot(file_path)
robot.show_robot_info()
