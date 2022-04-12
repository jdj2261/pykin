import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.objects.gripper import GripperManager
import pykin.utils.plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

file_path = '../../asset/urdf/panda/panda.urdf'
mesh_path = pykin_path+"/asset/urdf/panda/"
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))

configures = {}
configures["gripper_names"] = ["right_gripper", "leftfinger", "rightfinger"]
configures["gripper_max_width"] = 0.08
configures["gripper_max_depth"] = 0.035
configures["tcp_position"] = np.array([0, 0, 0.097])

gripper_manager = GripperManager(robot, mesh_path, **configures)
gripper_manager.set_eef_transform()
gripper_manager.visualize(ax, gripper_manager.gripper, color='r')
gripper_manager.set_tcp_transform()
gripper_manager.visualize(ax, gripper_manager.gripper, color='b')

plt.show_figure()