import numpy as np

from pykin.kinematics.transform import Transform
from pykin.robot import Robot

# If you want to check robot's collision, install python-fcl 
# and then, import FclManager in fcl_utils package
from pykin.utils.fcl_utils import FclManager
from pykin.utils.kin_utils import get_robot_geom
from pykin.utils import plot_utils as plt

file_path = '../asset/urdf/baxter/baxter.urdf'

robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

head_thetas = np.zeros(1)
right_arm_thetas = np.array([np.pi, 0, 0, 0, 0, 0, 0])
left_arm_thetas = np.array([-np.pi, 0, 0, 0, 0, 0, 0])

thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
transformations = robot.kin.forward_kinematics(thetas)

# call FclManager class
fcl_manager = FclManager()
for link, transformation in transformations.items():
    # get robot link's name and geometry info 
    name, gtype, gparam = get_robot_geom(robot.links[link])
    # get 4x4 size homogeneous transform matrix
    transform = transformation.matrix()
    # add link name, geometry info, transform matrix to fcl_manager 
    fcl_manager.add_object(name, gtype, gparam, transform)

# you can get collision result, contacted object name, fcl contatct_data
result, objs_in_collision, contact_data = fcl_manager.collision_check(return_names=True, return_data=True)

print(result, objs_in_collision, contact_data)