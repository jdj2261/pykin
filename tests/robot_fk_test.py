from pykin.robot import Robot
import pykin.robot

file_name = "../../asset/udrf/baxter.urdf"

robot = Robot(file_name)

head_thetas = [0.0]
right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

robot.get_desired_tree("base", "left_wrist")
fk = robot.forward_kinematics(left_arm_thetas)
