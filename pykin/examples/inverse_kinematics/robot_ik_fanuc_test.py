import numpy as np
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils
from pykin.utils import transform_utils as t_utils

file_path = "urdf/fanuc/fanuc_r2000ic_165f.urdf"
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("base_link", "link_6")

# target_thetas = [0, np.pi/2, 0, 0, 0, np.pi/4]
# target_thetas = [0.905, 1.879, -0.632, 0, 0, 0]
target_thetas = [0.785, 1.679, -0.638, 0.000, 2.101, -0.785]
target_thetas[1] = -(target_thetas[1] - np.pi/2)
# target_thetas[4] -= np.pi/2
init_thetas = np.random.randn(robot.arm_dof)

robot.set_transform(target_thetas)
_, ax = p_utils.init_3d_figure("FK", visible_axis=True)
p_utils.plot_robot(ax=ax, robot=robot, geom="visual", only_visible_geom=False, alpha=1)
end_pose = robot.get_info()['visual']['link_6'][-1]
print(end_pose)
print(np.rad2deg(t_utils.get_rpy_from_matrix(end_pose)))

fk = robot.forward_kin(target_thetas)
target_pose = robot.compute_eef_pose(fk)
joints = robot.inverse_kin(init_thetas, target_pose, method="LM")

print(joints)

robot.set_transform(joints)
_, ax = p_utils.init_3d_figure("IK", visible_axis=True)
p_utils.plot_robot(ax=ax, robot=robot, geom="visual", only_visible_geom=False, alpha=1)
p_utils.show_figure()
