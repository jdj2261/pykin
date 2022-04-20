import numpy as np
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.planners.cartesian_planner import CartesianPlanner
from pykin.utils.transform_utils import get_matrix_from_rpy
import pykin.utils.plot_utils as plt

def get_result_qpos(robot, init_qpos, eef_pos):
    is_limit_qpos = False
    result_qpos = robot.inverse_kin(init_qpos, eef_pos, method="LM")
    is_limit_qpos = robot.check_limit_joint(result_qpos)
    if is_limit_qpos:
        return result_qpos

    while not is_limit_qpos:
        result_qpos = robot.inverse_kin(np.random.randn(len(init_qpos)), eef_pos, method="LM")
        is_limit_qpos = robot.check_limit_joint(result_qpos)
    return result_qpos


fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")

custom_fpath = '../../asset/config/panda_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.06]))
green_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.12]))
support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

red_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
blue_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
green_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
box_goal_mesh = get_object_mesh('box_goal.stl', 0.001)
table_mesh = get_object_mesh('custom_table.stl', 0.01)

scene_mngr = SceneManager("collision", is_pyplot=True)
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="red_box", gtype="mesh", gparam=red_cube_mesh, h_mat=red_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="blue_box", gtype="mesh", gparam=blue_cube_mesh, h_mat=blue_box_pose.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="green_box", gtype="mesh", gparam=green_cube_mesh, h_mat=green_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=box_goal_mesh, h_mat=support_box_pose.h_mat, color=[1.0, 0, 1.0])
scene_mngr.add_robot(robot, init_qpos)

init_pose = scene_mngr.get_robot_eef_pose()
# print(init_pose)
grasp_pose = green_box_pose.h_mat
r_mat = get_matrix_from_rpy(np.array([0, np.pi/2, 0]))
grasp_pose[:3, :3] = r_mat
grasp_pose[:3, 3] = grasp_pose[:3, 3] - [0.1, 0, 0]

# target_thetas = scene_mngr.compute_ik(grasp_pose)
target_thetas = get_result_qpos(scene_mngr.robot, init_qpos, grasp_pose)


scene_mngr.set_robot_eef_pose(target_thetas)
scene_mngr.attach_object_on_gripper("green_box", False)

# scene_mngr.render_all_scene(ax, visible_geom=True, alpha=0.7)
# scene_mngr.show()


############################ Show collision info #############################
planner = RRTStarPlanner(
    delta_distance=0.1,
    epsilon=0.2, 
    gamma_RRT_star=2,
)

planner.run(
    scene_mngr=scene_mngr,
    cur_q=target_thetas, 
    goal_pose=init_pose,
    max_iter=1000)

# planner.simulate_planning(ax)

joint_path = planner.get_joint_path(n_step=10)

joint_trajectory = []
eef_poses = []
for step, joint in enumerate(joint_path):
    fk = robot.forward_kin(joint)
    joint_trajectory.append(joint)
    eef_poses.append(fk[robot.eef_name].pos)

# plt.plot_path_planner(ax, eef_poses)
# plt.show_figure()

# # TODO only scene_mngr
plt.plot_animation(
    scene_mngr,
    joint_trajectory, 
    fig, 
    ax,
    eef_poses=eef_poses,
    objects=scene_mngr.objs,
    geom=scene_mngr.geom,
    visible_objects=True,
    visible_geom=True, 
    interval=1, 
    repeat=True)