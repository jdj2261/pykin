import os, sys
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

import numpy as np

from pykin.kinematics import transform as tf
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("base", "right_l6")

# panda_example
target_thetas = [0, np.pi/2, 0, 0, 0, 0, 0, 0]
init_thetas = np.random.randn(7)

fk = robot.forward_kin(target_thetas)
_, ax = plt.init_3d_figure("Target Pose")
plt.plot_robot(robot, ax)

target_pose = robot.eef_pose
ik_result = robot.inverse_kin(init_thetas, target_pose, method="LM")

theta = np.concatenate((np.zeros(1), ik_result))
result_fk = robot.forward_kin(theta)

_, ax = plt.init_3d_figure("IK Result")
plt.plot_robot(robot, ax,
               visible_visual=False,
               mesh_path='../../asset/urdf/sawyer/')

err = robot.compute_pose_error(
    fk[robot.eef_name].homogeneous_matrix,
    result_fk[robot.eef_name].homogeneous_matrix)
print(err)

plt.show_figure()
