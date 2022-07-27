import numpy as np

from pykin.kinematics import transform as t_utils
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils

urdf_path = 'urdf/doosan/doosan_with_robotiq140.urdf'

robot = SingleArm(
    urdf_path, 
    t_utils.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.0]),
    has_gripper=True,
    gripper_name="robotiq140_gripper")

target_thetas = np.array([np.random.uniform(-np.pi, np.pi) for _ in range(robot.arm_dof)])
target_thetas = np.array([0, np.pi/3, 0, 0, 0, 0])
init_thetas = np.random.randn(robot.arm_dof)
robot.setup_link_name("base_0", "link6")
robot.set_transform(target_thetas)

robot.close_gripper(0.03)
_, ax = p_utils.init_3d_figure("Close Gripper", visible_axis=True)
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="collision",
               only_visible_geom=True,)

robot.open_gripper(0.03)
_, ax = p_utils.init_3d_figure("Open Gripper", visible_axis=True)
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="collision",
               only_visible_geom=True,)
p_utils.show_figure()