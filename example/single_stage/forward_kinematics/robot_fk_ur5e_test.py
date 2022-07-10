import numpy as np
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils

file_path = 'urdf/ur5e/ur5e.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("ur5e_base_link", "ur5e_right_hand")

target_thetas = [-0.470, -1.735, 2.480, -2.275, -1.590, -1.991]
robot.set_transform(target_thetas)

_, ax = p_utils.init_3d_figure("FK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True,
               alpha=1)
p_utils.show_figure()
