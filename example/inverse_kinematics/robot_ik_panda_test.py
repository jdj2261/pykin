
import numpy as np

from pykin.kinematics import transform as tf
from pykin.robot import Robot
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/panda/panda.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.set_desired_frame("panda_link0", "panda_hand")
print(robot.active_joint_names)
# panda_example
target_thetas = [0, np.pi/2, 0, 0, 0, 0, 0]
init_thetas = np.random.randn(7)

fk = robot.kin.forward_kinematics(target_thetas)
print(fk["panda_hand"].pos)
_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot, fk, ax, "panda")
ax.legend()

target_pose = np.concatenate((fk["panda_hand"].pos, fk["panda_hand"].rot))
ik_result, _ = robot.kin.inverse_kinematics(init_thetas, target_pose, method="LM")

theta = np.concatenate((ik_result, np.zeros(2)))
robot.reset_desired_frames()

fk = robot.kin.forward_kinematics(theta)
print(fk["panda_hand"].pos)
_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "panda", visible_visual=False,
               mesh_path='../asset/urdf/panda/')
ax.legend()
plt.show_figure()
