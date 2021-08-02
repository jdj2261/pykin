from robosuite import load_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.robots import Panda
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils import SimulationError
from robosuite.robots import SingleArm
from mujoco_py import MjSim, MjViewer, load_model_from_path, load_model_from_xml
import numpy as np

control_freq = 50
horizon = 10000
SIMULATION_TIMESTEP = 0.02
done = False
timestep = 0


def initialize_sim(xml_string=None, model=None):
    mjpy_model = load_model_from_xml(
        xml_string) if xml_string else model.get_model(mode="mujoco_py")
    sim = MjSim(mjpy_model)
    sim.forward()
    model_timestep, control_timestep = initialize_time(control_freq)
    return sim, model_timestep, control_timestep


def initialize_time(control_freq):

    model_timestep = SIMULATION_TIMESTEP
    if model_timestep <= 0:
        raise ValueError("Invalid simulation timestep defined!")

    if control_freq <= 0:
        raise SimulationError(
            "Control frequency {} is invalid".format(control_freq))
    control_timestep = 1. / control_freq

    return model_timestep, control_timestep


def step(sim, done, timestep, model_timestep, control_timestep, action):
    if done:
        raise ValueError("executing action in terminated episode")

    timestep += 1
    for i in range(int(control_timestep / model_timestep)):
        sim.forward()
        pre_action(sim, action)
        sim.step()

    done = post_action(timestep, action)
    return timestep, done


def pre_action(sim, action):
    sim.data.ctrl[:] = action


def post_action(timestep, action):
    done = (timestep >= horizon)
    return done


if __name__ == '__main__':


    # mujoco_arena = TableArena()
    # mujoco_arena.set_origin([0.8, 0, 0])
    mujoco_arena = MujocoXML(xml_path_completion("arenas/empty_arena.xml"))
    mujoco_robot = MujocoXML(xml_path_completion("robots/baxter/robot.xml"))
    mujoco_arena.merge(mujoco_robot)

    sim = MjSim(mujoco_arena.get_model())
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0

    for j in range(sim.model.njnt):
        print("Joint ID:", j, ",", "Joint Name:", sim.model.joint_id2name(
            j), ",", "Joint Limits", sim.model.jnt_range[j])

    sim.reset()
    sim.forward()

    for i in range(horizon):
        sim.data.ctrl[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sim.step()
        viewer.render()
