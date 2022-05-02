import numpy as np
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.place import PlaceAction
from pykin.action.pick import PickAction
import pykin.utils.plot_utils as plt

file_path = '../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")

file_path = '../../../asset/urdf/panda/panda.urdf'
panda_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[0, 0, 0]))
custom_fpath = '../../../asset/config/panda_init_params.yaml'
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

pick = PickAction(scene_mngr, n_contacts=5, n_directions=10)
place = PlaceAction(scene_mngr, n_samples_held_obj=3, n_samples_support_obj=5)

###### Surface sampling held and support obj#######
fig, ax = plt.init_3d_figure( name="Sampling Object")
support_points, _ = place.get_surface_points_for_support_obj("goal_box")
place.render_points(ax, support_points)
support_points, _ = place.get_surface_points_for_held_obj("green_box")
place.render_points(ax, support_points)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)

# ###### Get Release Pose #######
# eef_poses = list(pick.get_grasp_poses("green_box"))
# all_release_poses = []
# for eef_pose in eef_poses:
#     release_poses = list(place.get_release_poses("goal_box", "green_box", eef_pose))
#     for release_pose, obj_pose in release_poses:
#         fig, ax = plt.init_3d_figure( name="Get Release Pose")
#         all_release_poses.append((release_pose, obj_pose))
#         # place.render_axis(ax, release_pose, scale=0.05)
#         place.scene_mngr.render_gripper(ax, alpha=0.9, pose=release_pose)
#         place.scene_mngr.render.render_object(ax, place.scene_mngr.scene.objs["green_box"], obj_pose)
#         place.scene_mngr.render_objects(ax)
#         place.show()

###### Get Release Pose not Consider Gripper #######
# all_release_poses = list(place.get_release_poses("goal_box", "green_box"))
# for release_pose, obj_pose in all_release_poses:
#     fig, ax = plt.init_3d_figure( name="Get Release Pose")
#     place.scene_mngr.render.render_object(ax, place.scene_mngr.scene.objs["green_box"], obj_pose)
#     place.scene_mngr.render_objects(ax)
#     place.show()

# # # ###### Get Release Pose #######
fig, ax = plt.init_3d_figure( name="Get Release Pose")
tcp_poses = list(pick.get_tcp_poses("green_box"))
all_release_poses = []
for tcp_pose in tcp_poses:
    release_poses = list(place.get_release_poses("goal_box", "green_box", tcp_pose))
    for release_pose, obj_pose in release_poses:
        all_release_poses.append((release_pose, obj_pose))
        place.render_axis(ax, release_pose, scale=0.05)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)
print(len(all_release_poses))

##### Level wise - 1 #######
fig, ax = plt.init_3d_figure( name=f"Level wise 1")    
release_poses_for_only_gripper = list(place.get_release_poses_for_only_gripper(all_release_poses))
for release_pose_for_only_gripper, _ in release_poses_for_only_gripper:
    place.render_axis(ax, release_pose_for_only_gripper, scale=0.05)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)
print(len(release_poses_for_only_gripper))

###### Level wise - 2 #######
fig, ax = plt.init_3d_figure( name="Level wise 2")
goal_release_poses = list(place.get_release_poses_for_robot(release_poses_for_only_gripper))
for goal_release_pose, _ in goal_release_poses:
    place.render_axis(ax, goal_release_pose, scale=0.05)
print(len(goal_release_poses))

plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)
place.show()

