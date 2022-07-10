import numpy as np

from pykin.robots.single_arm import SingleArm
from pykin.kinematics import transform as t_utils
from pykin.utils import plot_utils as p_utils

urdf_path = 'urdf/iiwa7/iiwa7.urdf'

robot = SingleArm(urdf_path, t_utils.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

# iiwa7 example
target_thetas = np.array([np.random.uniform(-np.pi, np.pi) for _ in range(robot.arm_dof)])
init_thetas = np.random.randn(robot.arm_dof)
robot.setup_link_name("iiwa7_link_0", "iiwa7_right_hand")

robot.set_transform(target_thetas)
_, ax = p_utils.init_3d_figure("FK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True)

fk = robot.forward_kin(target_thetas)
target_pose = robot.compute_eef_pose(fk)
joints = robot.inverse_kin(init_thetas, target_pose, method="LM")

robot.set_transform(joints)
_, ax = p_utils.init_3d_figure("IK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True)
p_utils.show_figure()
