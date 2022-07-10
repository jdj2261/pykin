import numpy as np

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
from pykin.action.pick import PickAction
import pykin.utils.plot_utils as p_utils

file_path = 'urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])


bottle_meshes = []
for i in range(3):
    bottle_meshes.append(get_object_mesh('bottle.stl'))

bottle_pose1 = Transform(pos=np.array([0.6, 0.2, 0.74 + abs(bottle_meshes[0].bounds[0][2])]))
bottle_pose2 = Transform(pos=np.array([0.6, 0.35, 0.74 + abs(bottle_meshes[0].bounds[0][2])]))
bottle_pose3 = Transform(pos=np.array([0.6, 0.05, 0.74 + abs(bottle_meshes[0].bounds[0][2])]))

support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

goal_box_mesh = get_object_mesh('goal_box.stl', 0.001)
table_mesh = get_object_mesh('custom_table.stl', 0.01)

scene_mngr = SceneManager("collision", is_pyplot=True)
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="bottle1", gtype="mesh", gparam=bottle_meshes[0], h_mat=bottle_pose1.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="bottle2", gtype="mesh", gparam=bottle_meshes[1], h_mat=bottle_pose2.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="bottle3", gtype="mesh", gparam=bottle_meshes[2], h_mat=bottle_pose3.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=goal_box_mesh, h_mat=support_box_pose.h_mat, color=[1.0, 0, 1.0])
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["bottle1"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["bottle2"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["bottle3"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.logical_state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.logical_state.holding : None}
scene_mngr.update_logical_states()

pick = PickAction(scene_mngr, n_contacts=1, n_directions=1)

################# Action Test ##################
actions = list(pick.get_possible_actions_level_1())
fig, ax = p_utils.init_3d_figure(name="Level wise 1")
for pick_actions in actions:
    for all_grasp_pose in pick_actions[pick.info.GRASP_POSES]:
        pick.scene_mngr.render.render_axis(ax, all_grasp_pose[pick.move_data.MOVE_grasp])
        pick.scene_mngr.render.render_axis(ax, all_grasp_pose[pick.move_data.MOVE_pre_grasp])
        pick.scene_mngr.render.render_axis(ax, all_grasp_pose[pick.move_data.MOVE_post_grasp])
        pick.scene_mngr.render_gripper(ax, pose=all_grasp_pose[pick.move_data.MOVE_grasp])
pick.scene_mngr.render_objects(ax)
p_utils.plot_basis(ax)
pick.show()

# fig, ax = p_utils.init_3d_figure( name="Level wise 2")
# for pick_actions in actions:
#     for all_grasp_pose in pick_actions[pick.info.GRASP_POSES]:
#         ik_solve, grasp_pose = pick.get_possible_ik_solve_level_2(grasp_poses=all_grasp_pose)
#         if ik_solve is not None:
#             pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_grasp])
#             pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_pre_grasp])
#             pick.scene_mngr.render.render_axis(ax, grasp_pose[pick.move_data.MOVE_post_grasp])
            
# pick.scene_mngr.render_objects(ax)
# p_utils.plot_basis(ax)
# pick.show()