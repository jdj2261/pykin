import numpy as np

from pykin.robots.single_arm import SingleArm
from pykin.kinematics import transform as tf
from pykin.utils import plot_utils as plt

urdf_path = '../../asset/urdf/iiwa14/iiwa14.urdf'

robot = SingleArm(urdf_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# iiwa14 example
target_thetas = np.array([np.random.uniform(-np.pi, np.pi) for _ in range(robot.dof)])
init_thetas = np.random.randn(robot.dof)

robot.setup_link_name("iiwa14_link_0", "iiwa14_right_hand")
fk = robot.forward_kin(target_thetas)
target_pose = robot.get_eef_pose(fk)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot, ax, fk)

joints = robot.inverse_kin(init_thetas, target_pose, method="LM")
result_fk = robot.forward_kin(joints)

_, ax = plt.init_3d_figure("IK")
plt.plot_robot(
    robot, 
    ax=ax,
    transformations=result_fk)
plt.show_figure()
