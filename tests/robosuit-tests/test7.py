from robosuite import load_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.robots import Panda
from robosuite.models.base import MujocoXML
from robosuite.utils import SimulationError
from robosuite.robots import SingleArm
from mujoco_py import MjSim, MjViewer, load_model_from_path, load_model_from_xml
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, add_prefix

arena_model = MujocoXML(xml_path_completion("arenas/empty_arena.xml"))

# load baxter model
baxter_model = MujocoXML(xml_path_completion("robots/baxter/robot.xml"))
baxter_model.add_prefix("robot0_")


node = baxter_model.worldbody.find(
    "./body[@name='{}']".format("robot0_base"))
node.set("pos", array_to_string([-0.0, 0.0, 0.913]))

# load left gripper
left_gripper_model = MujocoXML(
     xml_path_completion("grippers/rethink_gripper.xml"))
left_gripper_model.add_prefix("gripper0_left_")

left_arm_subtree = baxter_model.worldbody.find(".//body[@name='robot0_left_hand']")
for body in left_gripper_model.worldbody:
    left_arm_subtree.append(body)
site = left_gripper_model.worldbody.find(
    ".//site[@name='{}']".format('gripper0_left_grip_site'))
site.set("rgba", "0 0 0 0")
site = left_gripper_model.worldbody.find(
    ".//site[@name='{}']".format('gripper0_left_grip_site_cylinder'))
site.set("rgba", "0 0 0 0")

# load right gripper
right_gripper_model = MujocoXML(
    xml_path_completion("grippers/rethink_gripper.xml"))
right_gripper_model.add_prefix("gripper0_right_")

right_arm_subtree = baxter_model.worldbody.find(".//body[@name='robot0_right_hand']")
for body in right_gripper_model.worldbody:
    right_arm_subtree.append(body)

site = right_gripper_model.worldbody.find(
    ".//site[@name='{}']".format('gripper0_right_grip_site'))
site.set("rgba", "0 0 0 0")
site = right_gripper_model.worldbody.find(
    ".//site[@name='{}']".format('gripper0_right_grip_site_cylinder'))
site.set("rgba", "0 0 0 0")

# merge XML
baxter_model.merge(left_gripper_model, merge_body=False)
baxter_model.merge(right_gripper_model, merge_body=False)
arena_model.merge(baxter_model)

# # # mjpy_model = model.get_model(mode="mujoco_py")

sim = MjSim(arena_model.get_model())
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0
while True:
    # action = np.random.randn(9)
    sim.data.ctrl[:] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sim.step()
    viewer.render()
