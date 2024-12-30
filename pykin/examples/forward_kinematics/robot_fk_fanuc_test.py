import numpy as np
import os

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as p_utils
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene

current_file_path = os.path.abspath(os.path.dirname(__file__))
file_path = "urdf/fanuc/fanuc_r2000ic_165f.urdf"
goal_qpos = np.array([np.pi/4, 0, np.pi/4, 0, np.pi/4, 0])

# Visual geometry 케이스
robot_visual = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
c_manager_visual = CollisionManager(is_robot=True)
c_manager_visual.setup_robot_collision(robot_visual, geom="visual")
robot_visual.setup_link_name("base_link", "link_6")
robot_visual.set_transform(goal_qpos)
fk_visual = robot_visual.forward_kin(goal_qpos)

_, ax = p_utils.init_3d_figure("FK", visible_axis=True)
p_utils.plot_robot(ax=ax, robot=robot_visual, geom="visual", only_visible_geom=False, alpha=1)
p_utils.show_figure()
