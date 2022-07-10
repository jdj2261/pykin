import numpy as np

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils

file_path = 'urdf/iiwa7/iiwa7.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

robot.setup_link_name(eef_name="iiwa7_right_hand")

target_thetas = [0, np.pi/3, 0, 0, 0, 0, 0]
robot.set_transform(target_thetas)

_, ax = p_utils.init_3d_figure("FK")
p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="collision",
               only_visible_geom=True)
p_utils.show_figure()
