import numpy as np

from pykin.kinematics import transform as tf
from pykin.robots.bimanual import Bimanual
from pykin.kinematics import jacobian as jac


file_path = '../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

left_arm_thetas = np.zeros(15)
robot.setup_link_name("base", "right_wrist")
robot.setup_link_name("base", "left_wrist")

fk = robot.forward_kin(left_arm_thetas)

J = {}
for arm in robot.arm_type:
    if robot.eef_name[arm]:
        J[arm] = jac.calc_jacobian(robot.desired_frames[arm], fk, len(np.zeros(7)))

print(J)