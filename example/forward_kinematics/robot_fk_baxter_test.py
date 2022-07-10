import numpy as np
from pykin.kinematics.transform import Transform
from pykin.robots.bimanual import Bimanual
from pykin.utils import plot_utils as p_utils

file_path = 'urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

target_thetas = np.zeros(robot.dof)
robot.set_transform(target_thetas)

_, ax = p_utils.init_3d_figure("FK")

p_utils.plot_robot(ax=ax, 
               robot=robot,
               geom="visual",
               only_visible_geom=True)
p_utils.show_figure()
