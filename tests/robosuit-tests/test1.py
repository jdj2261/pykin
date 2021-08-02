from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda, Baxter
from robosuite.models.arenas import TableArena
from mujoco_py import MjSim, MjViewer

world = MujocoWorldBase()

mujoco_robot = Baxter()
mujoco_robot.set_base_xpos([-0.5, 0, 0])
world.merge(mujoco_robot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0  # disable visualization of collision mesh

for i in range(10000):
    sim.data.ctrl[:] = [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    sim.step()
    viewer.render()
