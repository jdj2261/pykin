import numpy as np
import sys, os
import trimesh

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.tasks.grasp_old import GraspManager, GraspStatus
from pykin.utils.task_utils import get_relative_transform
from pykin.objects.object_manager import ObjectManager
import pykin.utils.plot_utils as plt

file_path = '../../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))
robot.setup_link_name(eef_name="panda_right_hand")

init_qpos = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, fk)

grasp_obj1_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
grasp_obj2_pose = Transform(pos=np.array([0.4, 0.2, 0.77]))
grasp_obj3_pose = Transform(pos=np.array([0.6, 0.1, 0.77]))

obs_pos2 = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
obs_pos3 = Transform(pos=np.array([0.4, 0.24, 0.0]))

cube_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/ben_cube.stl')
box_goal_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/box_goal.stl')
table_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/custom_table.stl')

cube_mesh.apply_scale(0.06)
box_goal_mesh.apply_scale(0.0013)
table_mesh.apply_scale(0.01)

object_man = ObjectManager()
object_man.add_object(name="red_box", gtype="mesh", gparam=cube_mesh, h_mat=grasp_obj1_pose.h_mat, for_grasp=True)
object_man.add_object(name="blue_box", gtype="mesh", gparam=cube_mesh, h_mat=grasp_obj2_pose.h_mat, for_grasp=True)
object_man.add_object(name="green_box", gtype="mesh", gparam=cube_mesh, h_mat=grasp_obj3_pose.h_mat, for_grasp=True)
object_man.add_object(name="box", gtype="mesh", gparam=box_goal_mesh, h_mat=obs_pos2.h_mat, for_support=True)
object_man.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=obs_pos3.h_mat)


o_manager = CollisionManager()

configures = {}
configures["gripper_names"] = ["right_gripper", "leftfinger", "rightfinger", "tcp"]
configures["gripper_max_width"] = 0.08
configures["gripper_max_depth"] = 0.035
configures["tcp_position"] = np.array([0, 0, 0.097])

#######################################
# 2. grasp test 
grasp_man = GraspManager(
    robot, 
    c_manager, 
    o_manager,
    mesh_path,    
    retreat_distance=0.15,
    release_distance=0.01,
    **configures)

###################################################################################
#generate_grasps
fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)
for i, (name, info) in enumerate(object_man.grasp_objects.items()):
    
    if i%3 == 0:
        color ='red'
    elif i%3 == 1:
        color = 'blue'
    else:
        color = 'green'
        
    grasp_object_info = object_man.get_info(name)
    support_obj_info = object_man.get_info(list(object_man.support_objects.keys())[0])
    grasp_waypoints = grasp_man.get_grasp_waypoints(
        obj_info=grasp_object_info, 
        limit_angle=0.05, 
        num_grasp=10, 
        n_trials=10)

    release_waypoints = grasp_man.get_release_waypoints(
        obj_info_on_sup=support_obj_info, 
        n_samples_on_sup=20, 
        obj_info_for_sup=grasp_object_info, 
        n_samples_for_sup=20, 
        n_trials=20)
    
    grasp_pose = grasp_waypoints[GraspStatus.grasp_pose]
    pre_grasp_pose = grasp_waypoints[GraspStatus.pre_grasp_pose]
    
    release_pose = release_waypoints[GraspStatus.release_pose]
    pre_release_pose = release_waypoints[GraspStatus.pre_release_pose]

    T = get_relative_transform(grasp_pose, info.h_mat)
    obj_grasp_pos_transformed = np.dot(pre_grasp_pose, T)
    obj_pre_release_pos_transformed = np.dot(pre_release_pose, T)
    obj_release_pos_transformed = np.dot(release_pose, T)

    grasp_man.visualize_axis(ax, grasp_pose, visible_basis=True)
    gripper = grasp_man.get_transformed_gripper_fk(grasp_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')
    
    # grasp_man.visualize_axis(ax, pre_grasp_pose, visible_basis=True)
    # gripper = grasp_man.get_transformed_gripper_fk(pre_grasp_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')

    grasp_man.visualize_axis(ax, release_pose, visible_basis=True)
    gripper = grasp_man.get_transformed_gripper_fk(release_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')

    # grasp_man.visualize_axis(ax, pre_release_pose, visible_basis=True)
    # gripper = grasp_man.get_transformed_gripper_fk(pre_release_pose, is_tcp=False)
    # grasp_man.visualize_gripper(ax, gripper, visible_basis=True, alpha=0.5, color='blue')

    # plt.plot_mesh(ax=ax, mesh=cube_mesh, h_mat=obj_grasp_pos_transformed, alpha=0.2, color=color)
    plt.plot_mesh(ax=ax, mesh=cube_mesh, h_mat=obj_release_pos_transformed, alpha=0.2, color=color)
    # plt.plot_mesh(ax=ax, mesh=cube_mesh, h_mat=obj_pre_release_pos_transformed, alpha=0.2, color=color)
    plt.plot_mesh(ax=ax, mesh=info.gparam, h_mat=info.h_mat, alpha=0.2, color=color)

result = grasp_man.object_c_manager.get_distances_internal()
for (o1, o2), distance in result.items():
    if o1 in list(grasp_man.object_c_manager.objects.grasp_objects.keys()) and o2 in list(grasp_man.object_c_manager.objects.grasp_objects.keys()):
        print(o1, o2, distance)

plt.plot_mesh(ax=ax, mesh=box_goal_mesh, h_mat=obs_pos2.h_mat, alpha=0.2)
plt.plot_mesh(ax=ax, mesh=table_mesh, h_mat=obs_pos3.h_mat, alpha=0.2)
plt.show_figure()