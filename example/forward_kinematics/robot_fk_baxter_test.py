import numpy as np

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import ShellColors as sc


# baxter_example
file_path = '../../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

head_thetas = [0.0]
right_arm_thetas = [np.pi/3, -np.pi/5, -np.pi/2, np.pi/7, 0, np.pi/7 ,0]
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

thetas = head_thetas + right_arm_thetas + left_arm_thetas
fk = robot.forward_kin(thetas)

for link, transform in fk.items():
    print(f"{sc.HEADER}{link}{sc.ENDC}, {transform.rot}, {transform.pos}")

_, ax = plt.init_3d_figure()
plt.plot_robot(
    robot,
    ax=ax,
    transformations=fk
)
plt.show_figure()

