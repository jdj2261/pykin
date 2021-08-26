import sys

from pykin.robot import Robot

file_path = '../asset/urdf/sawyer/sawyer.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
robot = Robot(file_path)


from pykin.utils import plot_utils as plt

fig, ax = plt.init_3d_figure()
plt.plot_robot(robot, 
               transformations=robot.transformations,
               ax=ax, 
               name=robot.robot_name,
               visible_visual=True, 
               visible_collision=False, 
               mesh_path='../asset/urdf/sawyer/')
ax.legend()
plt.show_figure()