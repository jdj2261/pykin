
import numpy as np
import pprint
from collections import OrderedDict
from dataclasses import dataclass

from pykin.robots.robot import Robot
from pykin.scene.object import Object
from pykin.scene.render import SceneRender
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import ShellColors as sc

@dataclass
class State:
    on = 'on'
    support = 'support'
    static = 'static'
    held = 'held'
    holding = 'holding'

class SceneManager:
    def __init__(self):
        # Element for Scene
        self.objs = OrderedDict()
        self.robot = None

        # Logical state
        self.state = State
        self.logical_states = OrderedDict()

        # Collision Manager
        self._obj_collision_mngr = CollisionManager()
        self._robot_collision_mngr = None
        self._gripper_collision_mngr = None

        # Render
        self.render = SceneRender()

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
        self._obj_collision_mngr.add_object(name, gtype, gparam, h_mat)

        # self.apply_object_logical_state(obj=self.objs[name])
        
    def add_robot(self, robot, fk=None):
        if self.robot is not None:
            raise ValueError(
                "robot {} already exists".format(robot.robot_name)
            )
        self.robot:Robot = robot

        self._robot_collision_mngr = CollisionManager(is_robot=True)
        self._robot_collision_mngr.setup_robot_collision(robot, fk)

        self._gripper_collision_mngr = CollisionManager()
        self._gripper_collision_mngr.setup_gripper_collision(robot, fk)
        
        # self.apply_gripper_logical_state(robot.gripper)

    def remove_object(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        
        self.objs.pop(name, None)
        self._obj_collision_mngr.remove_object(name)

    def attach_object_on_gripper(self):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

    def detach_object_from_gripper(self):
        if self.robot is None:
            raise ValueError("Robot needs to be added first")

    def get_object_pose(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))

    # TODO
    def set_object_pose(self, name, pose):
        pass

    def get_robot_eef_pose(self):
        pass

    def set_robot_eef_pose(self):
        pass

    def get_gripper_pose(self):
        pass

    def set_gripper_pose(self, pose=np.eye(4)):
        pass

    def get_gripper_tcp_pose(self):
        pass

    def set_gripper_tcp_pose(self, pose=np.eye(4)):
        pass

    def reset(self):
        pass

    def collides_obj_and_gripper(self):
        pass

    def collides_obj_and_robot(self):
        pass

    def collides_self_robot(self):
        pass

    def update_logical_states(self):
        for object_name, logical_state in self.logical_states.items():
            if logical_state.get(self.state.on):
                if not self.logical_states[logical_state[self.state.on].name].get(self.state.support):
                    self.logical_states[logical_state[self.state.on].name][self.state.support] = []
                if self.objs[object_name] not in self.logical_states[logical_state[self.state.on].name][self.state.support]:
                    self.logical_states[logical_state[self.state.on].name][self.state.support].append(self.objs[object_name])
            
            if logical_state.get(self.state.holding):
                self.logical_states[logical_state[self.state.holding].name][self.state.held] = True

    def print_scene_info(self):
        print(f"*"*23 + f" {sc.OKGREEN}Scene{sc.ENDC} "+ f"*"*23)
        pprint.pprint(self.objs)
        if self.robot:
            print(self.robot.robot_name, self.robot.offset)
            print(self.robot.gripper.name, self.robot.gripper.get_gripper_pose())
        print(f"*"*63 + "\n")

    def print_logical_states(self):
        print(f"*"*23 + f" {sc.OKGREEN}Logical States{sc.ENDC} "+ f"*"*23)
        pprint.pprint(self.logical_states)
        print(f"*"*63 + "\n")

    def render_all_scene(self, ax, alpha=0.3, robot_color=None):
        self.render.render_all_scene(ax, self.objs, self.robot, alpha, robot_color)

    def render_object_and_gripper(self, ax, alpha=0.3, gripper_color=None, visible_tcp=True):
        self.render.render_object_and_gripper(
            ax, 
            self.objs, 
            self.robot, 
            alpha, gripper_color, visible_tcp)

    def render_object(self, ax, alpha=0.3):
        self.render.render_object(ax, self.objs, alpha)

    def render_robot(self, ax, alpha=0.3, color=None):
        self.render.render_robot(ax, self.robot, alpha, color)

    def render_gripper(self, ax, alpha=0.3, gripper_color='b', visible_tcp=True):
        self.render.render_gripper(ax, self.robot, alpha, gripper_color, visible_tcp)

    @property
    def gripper_name(self):
        return self.robot.gripper.name