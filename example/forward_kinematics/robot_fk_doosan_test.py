import numpy as np
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils

file_path = 'urdf/doosan/doosan_with_robotiq140.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), has_gripper=True, gripper_name="robotiq140")
robot.setup_link_name("base_0", "right_hand")

target_thetas = np.array([0, np.pi/3, 0, 0, 0, 0])
robot.set_transform(target_thetas)

_, ax = p_utils.init_3d_figure("FK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True,
               alpha=1)
p_utils.show_figure()
