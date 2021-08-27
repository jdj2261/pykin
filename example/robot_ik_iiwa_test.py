import numpy as np

from pykin.robot import Robot
from pykin.kinematics import transform as tf
from pykin.utils import plot_utils as plt

file_path = '../asset/urdf/iiwa14/iiwa14.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# iiwa14 example
target_thetas = [0, 0, np.pi/2, 0, 0, 0, 0]
init_thetas = np.random.randn(7)

robot.set_desired_frame("iiwa_link_0", "iiwa_link_ee")
fk = robot.kin.forward_kinematics(target_thetas)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot, fk, ax, "iiwa14")
ax.legend()
plt.show_figure()

target_pose = np.concatenate((fk["iiwa_link_ee"].pos, fk["iiwa_link_ee"].rot))
ik_result, _ = robot.kin.inverse_kinematics(init_thetas, target_pose, method="LM")

robot.reset_desired_frames()
fk = robot.kin.forward_kinematics(ik_result)
_, ax = plt.init_3d_figure("IK")
plt.plot_robot(robot, fk, ax, "iiwa14", visible_visual=True,
               mesh_path='../asset/urdf/iiwa14/')
ax.legend()
plt.show_figure()
