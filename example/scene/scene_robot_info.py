import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

urdf_path = '../../asset/urdf/iiwa7/iiwa7.urdf'
robot = SingleArm(
    f_name=urdf_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name("iiwa7_link_0", "iiwa7_right_hand")

thetas = np.array([0, np.pi/3, 0, 0, 0, 0, 0])
robot.set_transform(thetas)

plt.plot_robot(ax, robot, "collision", visible_geom=True)
plt.show_figure()
