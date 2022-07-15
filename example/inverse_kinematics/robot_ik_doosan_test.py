import numpy as np

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils
from pykin.utils import transform_utils as t_utils

urdf_path = 'urdf/doosan/doosan_with_robotiq140.urdf'

robot = SingleArm(
    urdf_path, 
    Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.0]),
    has_gripper=True)

target_thetas = np.array([np.random.uniform(-np.pi, np.pi) for _ in range(robot.arm_dof)])
target_thetas = np.array([ 0, 0, np.pi/1.5, 0, np.pi/3,  0])
init_thetas = np.random.randn(robot.arm_dof)
robot.setup_link_name("base_0", "link6")

robot.set_transform(target_thetas)
_, ax = p_utils.init_3d_figure("FK", visible_axis=True)
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual")

fk = robot.forward_kin(target_thetas)
target_pose = robot.compute_eef_pose(fk)
# target_pose = [0.5,  0.,  .8,  2.60120314e-04, -1.77597821e-05, -9.99999966e-01,  1.38757856e-05]
# target_pose = t_utils.get_h_mat(position=target_pose[:3], orientation=target_pose[3:])
joints = robot.inverse_kin(init_thetas, target_pose, method="LM")
print(joints)

robot.set_transform(joints)
_, ax = p_utils.init_3d_figure("IK", visible_axis=True)
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               visible_text=True)
p_utils.show_figure()