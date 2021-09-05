import sys

from pykin.robot import Robot
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
robot = Robot(file_path)

fig, ax = plt.init_3d_figure("URDF")

# For Baxter robots, the name argument to the plot_robot function must be baxter.
plt.plot_robot(robot, 
               transformations=robot.transformations,
               ax=ax, 
               name=robot.robot_name,
               visible_visual=False, 
               visible_collision=False, 
               mesh_path='../../asset/urdf/baxter/')
ax.legend()
plt.show_figure()