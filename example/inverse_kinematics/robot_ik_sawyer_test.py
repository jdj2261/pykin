import numpy as np
import sys, os

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)
from pykin.kinematics import transform as tf
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt
from pykin.utils.transform_utils import compute_pose_error

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("sawyer_base", "sawyer_right_hand")

# panda_example
target_thetas = [0, np.pi/3, np.pi/2, np.pi/2, 0, np.pi/3, 0, 0]
init_thetas = np.random.randn(7)

fk = robot.forward_kin(target_thetas)
_, ax = plt.init_3d_figure("Target Pose")
plt.plot_robot(robot, ax, fk)

target_pose = robot.get_eef_pose(fk)
ik_result = robot.inverse_kin(init_thetas, target_pose, method="LM")

theta = np.concatenate((np.zeros(1), ik_result))
result_fk = robot.forward_kin(theta)

_, ax = plt.init_3d_figure("IK Result")
plt.plot_robot(robot, ax,result_fk)

err = compute_pose_error(
    fk[robot.eef_name].h_mat,
    result_fk[robot.eef_name].h_mat)
print(err)

print(result_fk[robot.eef_name].pose)

plt.show_figure()
