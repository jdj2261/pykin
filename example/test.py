import numpy as np
from pykin.robot import Robot
from pykin.kinematics.transform import Transform

# baxter_example
file_path = '../asset/urdf/baxter/baxter.urdf'
robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# set joints for targe pose
right_arm_thetas = np.random.randn(7)

# set init joints
init_right_thetas = np.random.randn(7)

# Before compute IK, you must set desired root and end link
robot.set_desired_frame("base", "right_wrist")

# Compute FK for target pose
target_fk = robot.kin.forward_kinematics(right_arm_thetas)

# get target pose
target_r_pose = np.hstack((target_fk["right_wrist"].pos, target_fk["right_wrist"].rot))

# Compute IK Solution using LM(Levenberg-Marquardt) or NR(Newton-Raphson) method
ik_right_result, _ = robot.kin.inverse_kinematics(init_right_thetas, target_r_pose, method="LM")

# Compare error btween Target pose and IK pose
result_fk = robot.kin.forward_kinematics(ik_right_result)
error = robot.compute_pose_error(
    target_fk["right_wrist"].matrix(),
    result_fk["right_wrist"].matrix())
print(error)