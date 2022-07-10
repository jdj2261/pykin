import numpy as np

from pykin.kinematics import transform as t_utils
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils

file_path = 'urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, t_utils.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("sawyer_base", "sawyer_right_hand")

target_thetas = [0, np.pi/3, 0, 0, 0, np.pi/3, 0, 0]
init_thetas = np.random.randn(7)

robot.set_transform(target_thetas)
_, ax = p_utils.init_3d_figure("FK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True)

fk = robot.forward_kin(target_thetas)
target_pose = robot.compute_eef_pose(fk)
joints = robot.inverse_kin(init_thetas, target_pose, method="LM")

joints = np.concatenate((np.zeros(1), joints))
robot.set_transform(joints)

_, ax = p_utils.init_3d_figure("IK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True)
p_utils.show_figure()
