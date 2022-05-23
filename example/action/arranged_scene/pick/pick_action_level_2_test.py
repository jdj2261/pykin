import numpy as np
import sys, os

pykin_path = os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
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

scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["red_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["blue_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["green_box"] = {scene_mngr.scene.state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.state.holding : None}
scene_mngr.update_logical_states()

pick = PickAction(scene_mngr, n_contacts=10, n_directions=10)

################# Action Test ##################
actions = list(pick.get_possible_actions_level_1())
fig, ax = plt.init_3d_figure(name="Level wise 1")
for pick_actions in actions:
    for all_grasp_pose in pick_actions[pick.action_info.GRASP_POSES]:
        pick.scene_mngr.render.render_axis(ax, all_grasp_pose[pick.move_data.MOVE_grasp])
        # pick.scene_mngr.render.render_axis(ax, all_grasp_pose[pick.move_data.MOVE_pre_grasp])
        # pick.scene_mngr.render.render_axis(ax, all_grasp_pose[pick.move_data.MOVE_post_grasp])
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)

fig, ax = plt.init_3d_figure( name="Level wise 2")
cnt = 0
for pick_actions in actions:
    for all_grasp_pose in pick_actions[pick.action_info.GRASP_POSES]: 
        ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=all_grasp_pose)
        if ik_solve is not None:
            cnt += 1
            pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_grasp])
            # pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_pre_grasp])
            # pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_post_grasp])
            
print(cnt)
pick.scene_mngr.render_objects(ax)
plt.plot_basis(ax)
pick.show()