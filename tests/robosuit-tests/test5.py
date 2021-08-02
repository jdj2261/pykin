from robosuite import load_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.robots import Panda
from robosuite.models.base import MujocoXML
from robosuite.utils import SimulationError
from robosuite.robots import SingleArm
from mujoco_py import MjSim, MjViewer, load_model_from_path, load_model_from_xml

if __name__ == '__main__':
    world = MujocoWorldBase()
    controller_config = load_controller_config(
        default_controller="JOINT_VELOCITY")
    controller_config["output_max"] = 1.0
    controller_config["output_min"] = -1.0
    control_freq = 50
    horizon = 1000

    mujoco_arena = TableArena()
    mujoco_arena.set_origin([0.8, 0, 0])

    robot = SingleArm(
        robot_type="Panda",
        idn=0,
        gripper_type=None,
        controller_config=controller_config,
        # initial_qpos=[0.0, 0.7, 0.0, -1.4, 0.0, -0.56, 0.0],
        # initialization_noise=robot_noise,
        # gripper_type="PandaGripper",
        control_freq=control_freq,
    )

    # print(robot.controller_config)
    robot.load_model()
    robot.robot_model.set_base_xpos([0, 0, 0])

    world.merge(mujoco_arena)
    world.merge(robot.robot_model)

    sim = MjSim(world.get_model())
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0

    sim.reset()
    robot.reset_sim(sim)
    robot.setup_references()
    robot.reset(deterministic=False)
    # sim.forward()

    for i in range(horizon):
        # action = np.random.randn(9)
        action = [1, 0, 0, 0, 0, 0, 0]

        policy_step = True
        # for i in range(int(control_timestep / model_timestep)):
        sim.forward()
        # pre_action(sim, action, policy_step)
        robot.control(action=action, policy_step=policy_step)
        print(robot.torques, robot._joint_positions, robot._joint_velocities)
        policy_step = False
        sim.step()

        viewer.render()
