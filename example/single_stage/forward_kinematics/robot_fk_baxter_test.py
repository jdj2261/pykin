import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)
from pykin.kinematics.transform import Transform
from pykin.robots.bimanual import Bimanual
from pykin.utils import plot_utils as plt

file_path = '../../../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

target_thetas = np.zeros(robot.dof)
robot.set_transform(target_thetas)

_, ax = plt.init_3d_figure("FK")

plt.plot_robot(ax=ax, 
               robot=robot,
               geom="collision",
               visible_geom=True)
plt.show_figure()
