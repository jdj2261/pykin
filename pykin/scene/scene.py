
import numpy as np
import pprint
from collections import OrderedDict
from dataclasses import dataclass


from pykin.scene.object import Object
from pykin.scene.render import RenderPyPlot, RenderTriMesh
from pykin.robots.single_arm import SingleArm
from pykin.collision.collision_manager import CollisionManager

from pykin.utils.kin_utils import ShellColors as sc, apply_robot_to_scene

@dataclass
class State:
    on = 'on'
    support = 'support'
    static = 'static'
    held = 'held'
    holding = 'holding'

class SceneManager:
    def __init__(self, geom="collision", render_pyplot=True):
        # Element for Scene
        self.geom = geom
        self.objs = OrderedDict()
        self.robot = None

        # Logical state
        self.state = State
        self.logical_states = OrderedDict()

        # Collision Manager
        self.obj_collision_mngr = CollisionManager()
        self.robot_collision_mngr = None
        self.gripper_collision_mngr = None

        # Render
        self.render_pyplot = render_pyplot
        if render_pyplot:
            self.render = RenderPyPlot()
        else:
            self.render = RenderTriMesh()
            

    def __repr__(self):
        return 'pykin.scene.scene.{}()'.format(type(self).__name__)

    def add_object(self, name, gtype, gparam, h_mat=None, color='k'):
        if name in self.objs:
            raise ValueError(
                "Duplicate name: object {} already exists".format(name)
            )

        if h_mat is None:
            h_mat = np.eye(4, dtype=np.float32)

        self.objs[name] = Object(name, gtype, gparam, h_mat, color)
        self.obj_collision_mngr.add_object(name, gtype, gparam, h_mat)

    def add_robot(self, robot, thetas=[]):
        if self.robot is not None:
            raise ValueError(
                "robot {} already exists".format(robot.robot_name)
            )
        self.robot:SingleArm = robot

        self.robot_collision_mngr = CollisionManager(is_robot=True)
        self.robot_collision_mngr.setup_robot_collision(robot, geom="collision")

        if self.robot.has_gripper:
            self.gripper_collision_mngr = CollisionManager()
            self.gripper_collision_mngr.setup_gripper_collision(robot)

        if np.array(thetas).size != 0:
            self.set_robot_eef_pose(thetas)
        
    def remove_object(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        
        self.objs.pop(name, None)
        self.obj_collision_mngr.remove_object(name)

    def attach_object_on_gripper(self, name, pose, only_gripper=True):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")
        
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))

        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        self.set_object_pose(name, pose)

        if not only_gripper:
            self.robot_collision_mngr.add_object(
                self.objs[name].name,
                self.objs[name].gtype,
                self.objs[name].gparam,
                pose)
        else:
            self.gripper_collision_mngr.add_object(
                self.objs[name].name,
                self.objs[name].gtype,
                self.objs[name].gparam,
                pose)

        self.obj_collision_mngr.remove_object(name)

    def detach_object_from_gripper(self, name, only_gripper=True):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

        if not only_gripper:
            self.robot_collision_mngr.remove_object(name)
        else:
            self.gripper_collision_mngr.remove_object(name)
        
        self.objs.pop(name, None)

    def get_object_pose(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))

        return self.objs[name].h_mat

    def set_object_pose(self, name, pose):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))

        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        self.objs[name].h_mat = pose
        self.obj_collision_mngr.set_transform(name, pose)

    def compute_ik(self, pose=np.eye(4), method="LM", max_iter=1000):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

        pose = np.asarray(pose)
        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        return self.robot.inverse_kin(
            current_joints=np.random.randn(self.robot.arm_dof),
            target_pose=pose,
            method=method,
            max_iter=max_iter)

    def get_robot_eef_pose(self):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

        return self.robot.info[self.geom][self.robot.eef_name][3]

    def set_robot_eef_pose(self, thetas):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")
        
        self.robot.set_transform(thetas)
        for link, info in self.robot.info["collision"].items():
            if link in self.robot_collision_mngr._objs:
                self.robot_collision_mngr.set_transform(link, info[3])

    def get_gripper_pose(self):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self.robot.gripper.get_gripper_pose()

    def set_gripper_pose(self, pose=np.eye(4)):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        self.robot.gripper.set_gripper_pose(pose)
        for link, info in self.robot.gripper.info.items():
            if link in self.gripper_collision_mngr._objs:
                self.gripper_collision_mngr.set_transform(link, info[3])

    def get_gripper_tcp_pose(self):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self.robot.gripper.get_gripper_tcp_pose()

    def set_gripper_tcp_pose(self, pose=np.eye(4)):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        self.robot.gripper.set_gripper_tcp_pose(pose)

    def collide_objs_and_robot(self, return_names=False):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")
        return self.robot_collision_mngr.in_collision_other(self.obj_collision_mngr, return_names)

    def collide_self_robot(self, return_names=False):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")
        return self.robot_collision_mngr.in_collision_internal(return_names)

    def collide_objs_and_gripper(self, return_names=False):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self.gripper_collision_mngr.in_collision_other(self.obj_collision_mngr, return_names)

    def update_logical_states(self):
        for object_name, logical_state in self.logical_states.items():
            if logical_state.get(self.state.on):
                if not self.logical_states[logical_state[self.state.on].name].get(self.state.support):
                    self.logical_states[logical_state[self.state.on].name][self.state.support] = []
                if self.objs[object_name] not in self.logical_states[logical_state[self.state.on].name][self.state.support]:
                    self.logical_states[logical_state[self.state.on].name][self.state.support].append(self.objs[object_name])
            
            if logical_state.get(self.state.holding):
                self.logical_states[logical_state[self.state.holding].name][self.state.held] = True

    def get_objs_info(self):
        return self.objs

    def get_robot_info(self):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")
        
        return self.robot.info

    def get_gripper_info(self):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
        
        return self.robot.gripper.info

    def show_scene_info(self):
        print(f"*"*23 + f" {sc.OKGREEN}Scene{sc.ENDC} "+ f"*"*23)
        pprint.pprint(self.objs)
        if self.robot:
            print(self.robot.robot_name, self.robot.offset)
        if self.robot.has_gripper:
            print(self.robot.gripper.name, self.robot.gripper.get_gripper_pose())
        print(f"*"*63 + "\n")

    def show_logical_states(self):
        print(f"*"*23 + f" {sc.OKGREEN}Logical States{sc.ENDC} "+ f"*"*23)
        pprint.pprint(self.logical_states)
        print(f"*"*63 + "\n")

    def render_all_scene(self, ax=None, alpha=0.3, robot_color=None):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

        if self.render_pyplot:
            self.render.render_all_scene(ax, self.objs, self.robot, self.geom, alpha, robot_color)
        else:
            self.render = RenderTriMesh()
            self.render.render_all_scene(objs=self.objs, robot=self.robot)

    def render_object_and_gripper(self, ax=None, alpha=0.3, robot_color=None, visible_tcp=True):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        if self.render_pyplot:
            self.render.render_object_and_gripper(
                ax, 
                self.objs, 
                self.robot, 
                alpha, robot_color, visible_tcp=visible_tcp)
        else:
            self.render = RenderTriMesh()
            self.render.render_object_and_gripper(objs=self.objs, robot=self.robot)

    def render_object(self, ax=None, alpha=0.3):
        if self.render_pyplot:
            self.render.render_object(ax, self.objs, alpha)
        else:
            self.render = RenderTriMesh()
            self.render.render_object(objs=self.objs)

    def render_robot(self, ax=None, alpha=0.3, color=None):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

        if self.render_pyplot:
            self.render.render_robot(ax, self.robot, self.geom, alpha, color)
        else:
            self.render = RenderTriMesh()
            self.render.render_robot(self.robot)

    def render_gripper(self, ax=None, alpha=0.3, robot_color='b', visible_tcp=True):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        if self.render_pyplot:
            self.render.render_gripper(ax, self.robot, alpha, robot_color, visible_tcp)
        else:
            self.render = RenderTriMesh()
            self.render.render_robot(self.robot)

    def show(self):
        self.render.show()

    def reset(self):
        self.obj_collision_mngr = None
        self.objs = OrderedDict()
        
        if self.robot is not None:
            self.robot_collision_mngr = None            
            self.robot = None
        if self.robot.has_gripper:
            self.gripper_collision_mngr = None
            self.robot.gripper = None

    @property
    def gripper_name(self):
        if not self.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
        
        return self.robot.gripper.name