from robosuite import load_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.robots import Panda
from robosuite.models.base import MujocoXML
from robosuite.utils import SimulationError
from robosuite.robots import SingleArm
from mujoco_py import MjSim, MjViewer, load_model_from_path, load_model_from_xml

control_freq = 50
horizon = 10000
SIMULATION_TIMESTEP = 0.02
done = False
timestep = 0

controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
controller_config["output_max"] = 1.0
controller_config["output_min"] = -1.0

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

def step(sim, done, robot, timestep, model_timestep, control_timestep, action):
    if done:
        raise ValueError("executing action in terminated episode")
    
    timestep += 1
    policy_step = True
    for i in range(int(control_timestep / model_timestep)):
        sim.forward()
        # pre_action(sim, action, policy_step)
        robot.control(action=action, policy_step=policy_step)
        policy_step = False
        sim.step()

    done = post_action(timestep, action)
    return timestep, done
    
def pre_action(sim, action):
    sim.data.ctrl[:] = action


def post_action(timestep, action):
    done = (timestep >= horizon)
    return done

def reset(sim):
    sim.reset()
    # reset_internel()
    sim.forward()
    # return self._get_observation()

# def reset_internel():
#     # reset robot
#     robot.setup_references()
#     robot.reset(deterministic=False)

#     # Setup sim time based on control frequency
#     self.cur_time = 0
#     self.timestep = 0
#     self.done = False

if __name__ == '__main__':

    world = MujocoWorldBase()
    cur_time = 0
    controller_config = load_controller_config(
        default_controller="JOINT_POSITION")
    # controller_config["output_max"] = 1.0
    # controller_config["output_min"] = -1.0
    robot_noise = {
        "magnitude": [0.05]*7,
        "type": "gaussian"
    }
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
    print(robot.controller_config)
    robot.load_model()
    robot.robot_model.set_base_xpos([0, 0, 0])
    # mujoco_robot = MujocoXML(xml_path_completion("robots/panda/robot.xml"))
    # mujoco_robot = MujocoXML(xml_path_completion("robots/panda/robot.xml"))
    # baxter_model = MujocoXML("franka_sim/franka_panda.xml")

    world.merge(mujoco_arena)
    world.merge(robot.robot_model)

    sim, model_timestep, control_timestep = initialize_sim(model=world)
    viewer = MjViewer(sim)
    viewer.vopt.geomgroup[0] = 0

    sim.reset()
    robot.reset_sim(sim)
    robot.setup_references()
    robot.reset(deterministic=False)
    sim.forward()
    for i in range(horizon):
        # action = np.random.randn(9)
        action = [0, 0, 0, 0, 0, 1, 0]
        
        if done:
            raise ValueError("executing action in terminated episode")

        timestep += 1
        policy_step = True
        for i in range(int(control_timestep / model_timestep)):
            sim.forward()
            # pre_action(sim, action, policy_step)
            robot.control(action=action, policy_step=policy_step)
            print(robot.torques)
            policy_step = False
            sim.step()

        # timestep, done = step(sim, done, robot, timestep, model_timestep, control_timestep, action)
        viewer.render()
        print(timestep, done)





    #.............................................

    # sim = MjSim(mujoco_arena.get_model())
    # # sim = MjSim(mujoco_arena.get_model())
    # viewer = MjViewer(sim)

    # print(sim.model.joint_names)
    # for j in range(sim.model.njnt):
    #     print("Joint ID:", j, ",", "Joint Name:", sim.model.joint_id2name(
    #         j), ",", "Joint Limits", sim.model.jnt_range[j])

    # def control(goal_joint_pos, sim, kp, kd):
    #     action = [0 for _ in range(sim.model.njnt)]
    #     # print(action, sim.model.njnt)
    #     current_joint_pos = sim.data.qpos
    #     current_joint_vel = sim.data.qvel

    #     for i in range(sim.model.njnt):
    #         action[i] = (goal_joint_pos[i] - current_joint_pos[i]) * kp
    #         - current_joint_vel[i]* kd
    #     return action


    # step = -1
    # kp = 2
    # kd = 1.2
    # sim_time = 2000
    # done = False
    # for t in range(sim_time):
    #     if done:
    #         break
    #     viewer.render()

    #     sim_state = sim.get_state()
    #     sim.data.ctrl[0] = 1
    #     sim.data.ctrl[1] = -1
    #     sim.data.ctrl[2] = 0
    #     sim.data.ctrl[3] = 1
    #     sim.data.ctrl[4] = 0
    #     sim.data.ctrl[5] = 1
    #     sim.data.ctrl[6] = 1
    #     # sim.data.ctrl[:] = [step, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     # sim.data.ctrl[:] = control(
    #     #     [0, -np.pi / 4, 0, 0, 0, 0, 0], sim, kp, kd
    #     # )
    #     # if t > 1200:
    #     #     action = control(
    #     #         [0, -np.pi / 4, 0, 0, 0, 0, 0], sim, kp, kd
    #     #     )
    #     # elif t > 800:
    #     #     action = control(
    #     #         [0, 0, 0, 0, 0, 0, 0], sim, kp, kd
    #     #     )
    #     # elif t > 400:
    #     #     action = control(
    #     #         [3 * np.pi / 2, 0, 0, 0, 0, 0, 0], sim, kp, kd
    #         # )

    #     sim.step()



# baxter_model = MujocoXML("franka_sim/franka_panda.xml")
# panda_model.set_base_xpos([-0.5, 0, 0])

# print(mujoco_robot.init_qpos)
# print(mujoco_robot.bodies)
# print(mujoco_robot.eef_name)
# print(mujoco_robot.joints)


# viewer.vopt.geomgroup[0] = 0

# body_id = sim.model.body_name2id('robot0_link7')
# x_joint_i = sim.model.get_joint_qpos_addr("robot0_joint1")
# y_joint_i = sim.model.get_joint_qpos_addr("robot0_joint2")

# print(body_id, x_joint_i, y_joint_i)

# jnt_idx = sim.model.get_joint
# jnt_idx= sim.model.joint_name2id(joint)
# n = len(jnt_idx)  # Franka is a 7-dof arm

# for i in range(n):
#     k = env.sim.model.jnt_bodyid[i]
#     print(env.sim.model.body_id2name(k))


