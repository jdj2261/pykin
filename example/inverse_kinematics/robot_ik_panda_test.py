import os, sys
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

import numpy as np

from pykin.kinematics import transform as tf
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/panda/panda.urdf'

robot = SingleArm(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("panda_link0", "panda_hand")

# panda_example
target_thetas = [0, np.pi/5, 0, 0, 0, 0, 0, 0, 0]
init_thetas = np.random.randn(7)

fk = robot.forward_kin(target_thetas)
_, ax = plt.init_3d_figure("Target Pose")
plt.plot_robot(robot, ax)

target_pose = robot.eef_pose
ik_result = robot.inverse_kin(init_thetas, target_pose, method="LM")
print(ik_result)
theta = np.concatenate((ik_result, np.zeros(2)))
result_fk = robot.forward_kin(theta)

_, ax = plt.init_3d_figure("IK Result")
plt.plot_robot(robot, ax,
               visible_visual=True,
               mesh_path='../../asset/urdf/panda/')
plt.show_figure()
