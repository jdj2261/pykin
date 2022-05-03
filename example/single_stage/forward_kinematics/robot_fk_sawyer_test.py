import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt

file_path = '../../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(
    rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name(eef_name="sawyer_right_hand")

target_thetas = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8, 0.0, 0]
robot.set_transform(target_thetas)

_, ax = plt.init_3d_figure("FK")

plt.plot_robot(robot=robot,
               ax=ax, 
               geom="collision",
               only_visible_geom=True)
plt.show_figure()
