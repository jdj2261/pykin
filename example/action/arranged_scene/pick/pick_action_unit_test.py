import numpy as np
import sys, os

pykin_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.pick import PickAction
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

pick = PickAction(scene_mngr, 5, 50)

###### All Contact Points #######
fig, ax = plt.init_3d_figure(name="Get contact points")
contact_points = list(pick.get_contact_points(obj_name="green_box"))
pick.scene_mngr.render.render_points(ax, contact_points)
pick.scene_mngr.render_objects(ax, alpha=0.5)
plt.plot_basis(ax)

##### All Grasp Pose #######
grasp_poses = list(pick.get_all_grasp_poses("green_box"))
fig, ax = plt.init_3d_figure(name="Get Grasp Pose")
for grasp_pose in grasp_poses:
    # pick.scene_mngr.render.render_axis(ax, grasp_pose["pre_grasp_pose"])
    pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.grasp_name.GRASP])
    # pick.scene_mngr.render.render_axis(ax, grasp_pose["post_grasp_pose"])
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
# pick.show()

# ###### Level wise - 1 #######
fig, ax = plt.init_3d_figure(name="Level wise 1")
grasp_poses_for_only_gripper = list(pick.get_all_grasp_poses_for_only_gripper(grasp_poses))
for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
    pick.scene_mngr.render.render_axis(ax, grasp_pose_for_only_gripper[pick.grasp_name.GRASP])
    pick.scene_mngr.render.render_axis(ax, grasp_pose_for_only_gripper[pick.grasp_name.PRE_GRASP])
    pick.scene_mngr.render.render_axis(ax, grasp_pose_for_only_gripper[pick.grasp_name.POST_GRASP])
    # pick.scene_mngr.render_gripper(ax, alpha=0.7, pose=grasp_pose_for_only_gripper[pick.grasp_name.PRE_GRASP])
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
# pick.show()

####### Level wise - 2 #######
fig, ax = plt.init_3d_figure(name="Level wise 2")
for grasp_pose_for_only_gripper in grasp_poses_for_only_gripper:
    thetas, grasp_pose = pick.compute_ik_solve_for_robot(grasp_pose=grasp_pose_for_only_gripper)
    if grasp_pose:
        pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.grasp_name.GRASP])
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
pick.show()
