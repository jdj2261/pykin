from robosuite.models import MujocoWorldBase
from robosuite.robots import SingleArm
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.utils.mjcf_utils import new_joint
from robosuite import load_controller_config

# typing
from robosuite.robots.robot import Robot
from robosuite.models.arenas import Arena
from robosuite.models.objects.generated_objects import MujocoGeneratedObject

from mujoco_py import MjSim, MjViewer
import gym
import numpy as np
from collections import OrderedDict
import glfw

control_freq: float = 50.0

controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
controller_config["output_max"] = 1.0
controller_config["output_min"] = -1.0

robot = SingleArm(
    robot_type="Panda",
    idn=0,
    # controller_config=controller_config,
    # initial_qpos=[0.0, 0.7, 0.0, -1.4, 0.0, -0.56, 0.0],
    # initialization_noise=robot_noise,
    gripper_type="PandaGripper",
    # gripper_visualization=True,
    # control_freq=control_freq,
)

robot.load_model()
robot.robot_model.set_base_xpos([0, 0, 0])

arena = TableArena()
arena.set_origin([0.8, 0, 0])

ball_obj = BallObject(
    name="ball",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()

box_obj = BoxObject(
    name="box",
    size_min=[0.015, 0.015, 0.015],  # [0.015, 0.015, 0.015],
    size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
    rgba=[0, 0.5, 0.5, 1]).get_obj()

ball_obj.set('pos', '1.0 0 1.0')


box_obj.set('pos', '0.8 0 1.0')

model = MujocoWorldBase()
model.merge(robot.robot_model)
model.merge(arena)
model.worldbody.append(ball_obj)
model.worldbody.append(box_obj)

mjpy_model = model.get_model(mode="mujoco_py")

sim = MjSim(mjpy_model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0
while True:
    # action = np.random.randn(9)
    sim.data.ctrl[:] = [1,0,0,0,0,0,0,0,0]
    sim.step()
    viewer.render()
