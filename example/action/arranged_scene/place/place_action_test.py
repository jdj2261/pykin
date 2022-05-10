import numpy as np
import sys, os

pykin_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.pick import PickAction
from pykin.action.place import PlaceAction
import pykin.utils.plot_utils as plt

file_path = '../../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

file_path = '../../../../asset/urdf/panda/panda.urdf'
panda_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[0, 0, 0]))

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.35, 0.77]))
green_box_pose = Transform(pos=np.array([0.6, 0.05, 0.77]))
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
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["red_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["blue_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["red_box"]}
scene_mngr.scene.logical_states["green_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["blue_box"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.state.holding : None}
scene_mngr.update_logical_states()

pick = PickAction(scene_mngr, n_contacts=2, n_directions=10)
place = PlaceAction(scene_mngr, n_samples_held_obj=2, n_samples_support_obj=1)

###### Surface sampling held and support obj#######
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120, name="Sampling Object")
support_points, _ = place.get_surface_points_for_support_obj("goal_box")
place.render_points(ax, support_points)
support_points, _ = place.get_surface_points_for_held_obj("green_box")
place.render_points(ax, support_points)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax, alpha=0.5)

##### All Release Pose #######
fig, ax = plt.init_3d_figure( name="Get Release Pose")
eef_poses = list(pick.get_all_grasp_poses("green_box"))
all_release_poses = []
for eef_pose in eef_poses:
    release_poses = list(place.get_all_release_poses("goal_box", "green_box", eef_pose[pick.grasp_name.GRASP]))
    for release_pose, obj_pose in release_poses:
        all_release_poses.append((release_pose, obj_pose))
        pick.render_axis(ax, release_pose[place.release_name.RELEASE])
        # pick.render_axis(ax, release_pose[place.release_name.PRE_RELEASE])
        # pick.render_axis(ax, release_pose[place.release_name.POST_RELEASE])
        place.scene_mngr.render.render_object(ax, place.scene_mngr.scene.objs["green_box"], obj_pose)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)

# ###### Level wise - 1 #######
fig, ax = plt.init_3d_figure(name="Level wise 1")
release_poses_for_only_gripper = list(place.get_release_poses_for_only_gripper(all_release_poses, False))
for release_pose_for_only_gripper, obj_pose in release_poses_for_only_gripper:
    place.render_axis(ax, release_pose_for_only_gripper[place.release_name.RELEASE], scale=0.05)
    # place.scene_mngr.render_gripper(ax, pose=release_pose_for_only_gripper[place.release_name.RELEASE])
    place.scene_mngr.render.render_object(ax, place.scene_mngr.scene.objs["green_box"], obj_pose)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)

# ###### Level wise - 2 #######
fig, ax = plt.init_3d_figure(name="Level wise 2")
release_poses_for_only_gripper = list(place.get_release_poses_for_only_gripper(all_release_poses, False))
for release_pose_for_only_gripper, obj_pose in release_poses_for_only_gripper:
    ik_sol, release_pose = place.compute_ik_solve_for_robot(release_pose=release_pose_for_only_gripper, is_attach=False)
    
    if ik_sol:
        place.render_axis(ax, release_pose_for_only_gripper[place.release_name.RELEASE], scale=0.05)
        # place.scene_mngr.render_gripper(ax, pose=release_pose_for_only_gripper[place.release_name.RELEASE])
        place.scene_mngr.render.render_object(ax, place.scene_mngr.scene.objs["green_box"], obj_pose)
plt.plot_basis(ax)
place.scene_mngr.render_objects(ax)

place.show()