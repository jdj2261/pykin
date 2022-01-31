import numpy as np
import argparse
import sys, os
import yaml
import trimesh

parent_dir = os.path.dirname(os.getcwd())
pykin_path = parent_dir + "/../../"
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.planners.cartesian_planner import CartesianPlanner
from pykin.collision.collision_manager import CollisionManager
from pykin.utils import plot_utils as plt
from pykin.utils.object_utils import ObjectManager


help_str = "python panda_cartesian_planning.py"\
            " --timesteps 100 --damping 0.03 --resolution 0.2 --pos-sensitivity 0.03"

parser = argparse.ArgumentParser(usage=help_str)
parser.add_argument("--timesteps", type=int, default=500)
parser.add_argument("--damping", type=float, default=0.03)
parser.add_argument("--resolution", type=float, default=0.05)
parser.add_argument("--pos-sensitivity", type=float, default=0.04)
args = parser.parse_args()

file_path = pykin_path+'asset/urdf/panda/panda.urdf'
mesh_path = pykin_path+"/asset/urdf/panda/"
yaml_fpath = pykin_path+'/asset/config/panda_init_params.yaml'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("panda_link_0", "panda_right_hand")

with open(yaml_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
fk = robot.forward_kin(np.array(init_qpos))

init_eef_pose = robot.get_eef_pose(fk)
goal_eef_pose = controller_config["goal_pos"]
##################################################################

c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, fk, geom="collision")

c_manager.show_collision_info()

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
milk_mesh = trimesh.load_mesh(milk_path)

obs = ObjectManager()
o_manager = CollisionManager(milk_path)
name="milk1"
obs_pos=[3.73820701e-01, -2.51826813e-01,  2.71833382e-01]

obs(name=name, gtype="mesh", gparam=milk_mesh, h_mat=Transform(pos=obs_pos).h_mat)
obs(name="milk2", gtype="mesh", gparam=milk_mesh, h_mat=Transform(pos=[4.18720325e-01, -5.76662613e-02,  2.94687778e-01]).h_mat)
# o_manager.setup_object_collision(obs)

task_plan = CartesianPlanner(
    robot, 
    n_step=args.timesteps,
    dimension=7,
    damping=args.damping,
    pos_sensitivity=args.pos_sensitivity)

joint_path, target_poses = task_plan.get_path_in_joinst_space(
    cur_q=init_qpos,
    goal_pose=goal_eef_pose,
    robot_col_manager=c_manager,
    object_col_manager=o_manager,
    resolution=args.resolution)

if joint_path is None and target_poses is None:
    print("Cannot Visulization Path")
    exit()

joint_trajectory = []
for joint in joint_path:
    transformations = robot.forward_kin(joint)
    joint_trajectory.append(transformations)

print(f"Computed Goal Position : {joint_trajectory[-1][robot.eef_name].pose}")
print(f"Desired Goal position : {target_poses[-1]}")

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

plt.plot_animation(
    robot,
    joint_trajectory,
    fig=fig, 
    ax=ax,
    visible_collision=True,
    eef_poses=target_poses,
    objects=obs,
    visible_objects=True,
    repeat=True)
