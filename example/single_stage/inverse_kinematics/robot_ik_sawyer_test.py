import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics import transform as tf
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt
from pykin.utils.transform_utils import compute_pose_error

file_path = '../../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("sawyer_base", "sawyer_right_hand")

target_thetas = [0, np.pi/3, 0, 0, 0, np.pi/3, 0, 0]
init_thetas = np.random.randn(7)

robot.set_transform(target_thetas)
_, ax = plt.init_3d_figure("FK")
plt.plot_robot(ax=ax, 
               robot=robot,
               geom="collision",
               visible_geom=True)

fk = robot.forward_kin(target_thetas)
target_pose = robot.compute_eef_pose(fk)
joints = robot.inverse_kin(init_thetas, target_pose, method="LM")

joints = np.concatenate((np.zeros(1), joints))
robot.set_transform(joints)

_, ax = plt.init_3d_figure("IK")
plt.plot_robot(ax=ax, 
               robot=robot,
               geom="collision",
               visible_geom=True)
plt.show_figure()
