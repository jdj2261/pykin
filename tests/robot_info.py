import pykin.robot
from pykin.robot import Robot

file_name = '../asset/urdf/baxter.urdf'

robot = Robot(file_name)
robot.show_robot_info()

print(robot.joints)
print(robot.links)
print(robot.tree.root)
print(robot.num_links)
print(robot.num_active_joints)
print(robot.get_active_joint_names)