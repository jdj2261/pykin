import numpy as np

from pykin.kinematics import transform as tf
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/panda/panda.urdf'

robot = SingleArm(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.0]))
robot.setup_link_name("panda_link0", "panda_right_hand")

#panda_example
target_thetas = np.array([0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0])
init_thetas = np.random.randn(7)

fk = robot.forward_kin(target_thetas)
_, ax = plt.init_3d_figure("Target Pose")
plt.plot_robot(robot, ax, fk)

target_pose = robot.get_eef_pose(fk)
ik_result = robot.inverse_kin(init_thetas, target_pose, method="LM")
print(ik_result)
result_fk = robot.forward_kin(ik_result)

_, ax = plt.init_3d_figure("IK Result")
plt.plot_robot(robot, ax, fk)
plt.show_figure()
