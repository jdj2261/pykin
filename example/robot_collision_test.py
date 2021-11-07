import numpy as np

from pykin.kinematics.transform import Transform
from pykin.robots.bimanual import Bimanual
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.collision_utils import get_robot_collision_geom
from pykin.utils import plot_utils as plt

file_path = '../asset/urdf/baxter/baxter.urdf'

robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

head_thetas = np.zeros(1)
right_arm_thetas = np.array([np.pi, 0, 0, 0, 0, 0, 0])
left_arm_thetas = np.array([-np.pi, 0, 0, 0, 0, 0, 0])

thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
transformations = robot.forward_kin(thetas)

collision_manager = CollisionManager()
for link, transformation in transformations.items():
    name, gtype, gparam = get_robot_collision_geom(robot.links[link])
    transform = transformation.h_mat
    collision_manager.add_object(name, gtype, gparam, transform)

result, objs_in_collision, contact_data = collision_manager.collision_check(return_names=True, return_data=True)
print(result, objs_in_collision, contact_data)

fig, ax = plt.init_3d_figure()
plt.plot_robot(robot, ax, transformations, visible_collision=True)

left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
transformations = robot.forward_kin(thetas)

for link, transformation in transformations.items():
    name, _, _ = get_robot_collision_geom(robot.links[link])
    transform = transformation.h_mat
    collision_manager.set_transform(name=name, transform=transform)

result, objs_in_collision, contact_data = collision_manager.collision_check(return_names=True, return_data=True)
print(result, objs_in_collision, contact_data)

# fig, ax = plt.init_3d_figure()
# plt.plot_robot(robot, ax, transformations, visible_collision=True)
# plt.show_figure()