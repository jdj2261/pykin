import numpy as np
import pprint
import matplotlib.animation as animation
from collections import OrderedDict
from dataclasses import dataclass
from copy import deepcopy

from pykin.scene.object import Object
from pykin.scene.render import RenderPyPlot, RenderTriMesh
from pykin.robots.single_arm import SingleArm
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.action_utils import get_relative_transform
from pykin.utils.kin_utils import ShellColors as sc

@dataclass
class State:
    on = 'on'
    support = 'support'
    static = 'static'
    held = 'held'
    holding = 'holding'

class Scene:
    def __init__(self):
        self.objs = OrderedDict()
        self.robot:SingleArm = None
        self.logical_states = OrderedDict()
        self.state = State

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

    def update_logical_states(self):
        for object_name, logical_state in self.logical_states.items():
            if logical_state.get(State.on):
                if not self.logical_states[logical_state[State.on].name].get(State.support):
                    self.logical_states[logical_state[State.on].name][State.support] = []
                if self.objs[object_name] not in self.logical_states[logical_state[State.on].name][State.support]:
                    self.logical_states[logical_state[State.on].name][State.support].append(self.objs[object_name])
            
            if logical_state.get(State.holding):
                self.logical_states[logical_state[State.holding].name][State.held] = True


class SceneManager:
    def __init__(self, geom="collision", is_pyplot=True, scene=None):
        # Element for Scene
        self.geom = geom

        self.scene = scene
        if scene is None:
            self.scene = Scene()

        # Collision Manager
        self.obj_collision_mngr = CollisionManager()
        self.robot_collision_mngr = None
        self.gripper_collision_mngr = None

        # Render
        self.is_pyplot = is_pyplot
        if is_pyplot:
            self.render = RenderPyPlot()
        else:
            self.render = RenderTriMesh()

        # Attach / Detach
        self.is_attached = False
            
    def __repr__(self):
        return 'pykin.scene.scene.{}()'.format(type(self).__name__)

    def add_object(self, name, gtype, gparam, h_mat=None, color='k'):
        if name in self.scene.objs:
            raise ValueError(
                "Duplicate name: object {} already exists".format(name)
            )

        if h_mat is None:
            h_mat = np.eye(4, dtype=np.float32)

        self.scene.objs[name] = Object(name, gtype, gparam, h_mat, color)
        self.obj_collision_mngr.add_object(name, gtype, gparam, h_mat)

    def add_robot(self, robot:SingleArm, thetas=[]):
        if self.scene.robot is not None:
            raise ValueError(
                "robot {} already exists".format(robot.robot_name)
            )
        self.scene.robot = robot
        
        if np.array(thetas).size != 0:
            self.scene.robot.set_transform(thetas)

        self.robot_collision_mngr = CollisionManager(is_robot=True)
        self.robot_collision_mngr.setup_robot_collision(robot, geom=self.geom)

        if self.scene.robot.has_gripper:
            self.gripper_collision_mngr = CollisionManager()
            self.gripper_collision_mngr.setup_gripper_collision(robot)
        
    def remove_object(self, name):
        if name not in self.scene.objs:
            raise ValueError("object {} needs to be added first".format(name))
        
        self.scene.objs.pop(name, None)
        self.obj_collision_mngr.remove_object(name)

    def attach_object_on_gripper(self, name, only_gripper=False):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")
        
        if name not in self.scene.objs:
            raise ValueError("object {} needs to be added first".format(name))

        self.is_attached = True
        self.scene.robot.gripper.is_attached = self.is_attached
        self.attached_obj_name = self.scene.objs[name].name
        self.scene.robot.gripper.attached_obj_name = self.scene.objs[name].name
        
        self.obj_collision_mngr.remove_object(name)
        
        eef_pose = self.get_gripper_pose()
        self._transform_bet_gripper_n_obj = get_relative_transform(eef_pose, self.scene.objs[name].h_mat)

        if not only_gripper:
            self.robot_collision_mngr.add_object(
                self.scene.objs[name].name,
                self.scene.objs[name].gtype,
                self.scene.objs[name].gparam,
                self.scene.objs[name].h_mat)
            self.scene.robot.info["collision"][name] = [self.scene.objs[name].name, self.scene.objs[name].gtype, self.scene.objs[name].gparam, self.scene.objs[name].h_mat]
            self.scene.robot.info["visual"][name] = [self.scene.objs[name].name, self.scene.objs[name].gtype, self.scene.objs[name].gparam, self.scene.objs[name].h_mat]

        # TODO [gripper_collision_mngr이 필요한가??]
        self.gripper_collision_mngr.add_object(
            self.scene.objs[name].name,
            self.scene.objs[name].gtype,
            self.scene.objs[name].gparam,
            self.scene.objs[name].h_mat)       
        self.scene.robot.gripper.info[name] = [self.scene.objs[name].name, self.scene.objs[name].gtype, self.scene.objs[name].gparam, self.scene.objs[name].h_mat]

        self.scene.objs.pop(name, None)

    def detach_object_from_gripper(self, only_gripper=False):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")

        if not only_gripper:
            self.robot_collision_mngr.remove_object(self.attached_obj_name)
            self.scene.robot.info["collision"].pop(self.attached_obj_name)
            self.scene.robot.info["visual"].pop(self.attached_obj_name)

        self.gripper_collision_mngr.remove_object(self.attached_obj_name)
        self.scene.robot.gripper.info.pop(self.attached_obj_name)

        self.is_attached = False
        self.scene.robot.gripper.is_attached = False
        self.scene.robot.gripper.attached_obj_name = None

    def get_object_pose(self, name):
        if name not in self.scene.objs:
            raise ValueError("object {} needs to be added first".format(name))

        return self.scene.objs[name].h_mat

    def set_object_pose(self, name, pose):
        if name not in self.scene.objs:
            raise ValueError("object {} needs to be added first".format(name))

        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        self.scene.objs[name].h_mat = pose
        self.obj_collision_mngr.set_transform(name, pose)

    def compute_ik(self, pose=np.eye(4), method="LM", max_iter=100):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")

        pose = np.asarray(pose)
        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        return self.scene.robot.inverse_kin(
            current_joints=np.random.randn(self.scene.robot.arm_dof),
            target_pose=pose,
            method=method,
            max_iter=max_iter)

    def get_robot_eef_pose(self):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")

        return self.scene.robot.info[self.geom][self.scene.robot.eef_name][3]

    def set_robot_eef_pose(self, thetas):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")

        self.scene.robot.set_transform(thetas)
        for link, info in self.scene.robot.info[self.geom].items():
            if link in self.robot_collision_mngr._objs:
                self.robot_collision_mngr.set_transform(link, info[3])
            
            if self.scene.robot.has_gripper:
                if link in self.gripper_collision_mngr._objs:
                    self.gripper_collision_mngr.set_transform(link, info[3])

        if self.is_attached:
            self.scene.robot.info["collision"][self.attached_obj_name][3] = np.dot(self.get_gripper_pose(), self._transform_bet_gripper_n_obj)
            self.scene.robot.info["visual"][self.attached_obj_name][3] = np.dot(self.get_gripper_pose(), self._transform_bet_gripper_n_obj)
            self.scene.robot.gripper.info[self.attached_obj_name][3] = np.dot(self.get_gripper_pose(), self._transform_bet_gripper_n_obj)

    def get_gripper_pose(self):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self.scene.robot.gripper.get_gripper_pose()

    def set_gripper_pose(self, pose=np.eye(4)):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        self.scene.robot.gripper.set_gripper_pose(pose)
        for link, info in self.scene.robot.gripper.info.items():
            if link in self.gripper_collision_mngr._objs:
                self.gripper_collision_mngr.set_transform(link, info[3])

    def get_gripper_tcp_pose(self):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self.scene.robot.gripper.get_gripper_tcp_pose()

    def set_gripper_tcp_pose(self, pose=np.eye(4)):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        self.scene.robot.gripper.set_gripper_tcp_pose(pose)
        for link, info in self.scene.robot.gripper.info.items():
            if link in self.gripper_collision_mngr._objs:
                self.gripper_collision_mngr.set_transform(link, info[3])

    def collide_objs_and_robot(self, return_names=False):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")
        return self.robot_collision_mngr.in_collision_other(self.obj_collision_mngr, return_names)

    def collide_self_robot(self, return_names=False):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")
        return self.robot_collision_mngr.in_collision_internal(return_names)

    def collide_objs_and_gripper(self, return_names=False):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
    
        return self.gripper_collision_mngr.in_collision_other(self.obj_collision_mngr, return_names)

    def update_logical_states(self):
        self.scene.update_logical_states()

    def get_objs_info(self):
        return self.scene.objs

    def get_robot_info(self):
        if self.scene.robot is None:
            raise ValueError("Robot needs to be added first")
        
        return self.scene.robot.info

    def get_gripper_info(self):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
        
        return self.scene.robot.gripper.info

    def show_scene_info(self):
        self.scene.show_scene_info()

    def show_logical_states(self):
        self.scene.show_logical_states()

    def render_scene(
        self, 
        ax=None,
        scene=None,
        alpha=0.3, 
        robot_color=None,
        visible_geom=True,
        visible_text=False
    ):
        scene = scene
        if scene is None:
            scene = self.scene
            
        if scene.robot is None:
            raise ValueError("Robot needs to be added first")

        if self.is_pyplot:
            self.render.render_scene(
                ax, 
                scene.objs, 
                scene.robot, 
                alpha, 
                robot_color, 
                geom=self.geom, 
                visible_geom=visible_geom,
                visible_text=visible_text)
        else:
            self.render = RenderTriMesh()
            self.render.render_scene(objs=scene.objs, robot=scene.robot, geom=self.geom)
            
    def render_objects_and_gripper(
        self, 
        ax=None, 
        scene=None,
        alpha=0.3, 
        robot_color=None, 
        visible_tcp=True
    ):
        scene = scene
        if scene is None:
            scene = self.scene

        if not scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        if self.is_pyplot:
            self.render.render_objects_and_gripper(
                ax, 
                scene.objs, 
                scene.robot, 
                alpha, robot_color, visible_tcp=visible_tcp)
        else:
            self.render = RenderTriMesh()
            self.render.render_objects_and_gripper(objs=scene.objs, robot=scene.robot)

    def render_objects(self, ax=None, scene=None, alpha=0.3):
        scene = scene
        if scene is None:
            scene = self.scene

        if self.is_pyplot:
            self.render.render_objects(ax, scene.objs, alpha)
        else:
            self.render = RenderTriMesh()
            self.render.render_objects(objs=scene.objs)

    def render_robot(
        self, 
        ax=None, 
        scene=None,
        alpha=0.3, 
        robot_color=None,
        visible_geom=True,
        visible_text=False
    ):
        scene = scene
        if scene is None:
            scene = self.scene

        if scene.robot is None:
            raise ValueError("Robot needs to be added first")

        if self.is_pyplot:
            self.render.render_robot(
                ax, scene.robot, alpha, robot_color, self.geom, visible_geom, visible_text)
        else:
            self.render = RenderTriMesh()
            self.render.render_robot(scene.robot, self.geom)

    def render_gripper(
        self, 
        ax=None, 
        scene=None,
        alpha=0.3, 
        robot_color=None, 
        visible_tcp=True, 
        pose=None,
        only_visible_axis=False,
    ):
        scene = scene
        if scene is None:
            scene = self.scene

        if not scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        if self.is_pyplot:
            self.render.render_gripper(
                ax=ax, 
                robot=scene.robot, 
                alpha=alpha, 
                color=robot_color, 
                visible_tcp=visible_tcp,
                pose=pose,
                only_visible_axis=only_visible_axis)
        else:
            self.render = RenderTriMesh()
            self.render.render_gripper(scene.robot)

    def animation(
        self,
        ax=None, 
        fig=None,
        scene=None,
        alpha=0.3, 
        robot_color=None,
        joint_path=[], 
        eef_poses=[], 
        visible_geom=True,
        visible_text=True,
        interval=1,
        repeat=True
    ):
        if not self.is_pyplot:
            ValueError("Only pyplot can render.")
        
        scene = scene
        if scene is None:
            scene = self.scene

        def update(i):
            if i == len(joint_path)-1:
                print("Animation Finished..")
            ax.clear()

            if scene.objs:
                self.render.render_objects(ax, scene.objs, 0.3)
            
            if eef_poses is not None:
                self.render.render_trajectory(ax, eef_poses)
            
            self.set_robot_eef_pose(joint_path[i])
            self.render.render_robot(
                ax=ax,
                robot=scene.robot,
                alpha=alpha,
                color=robot_color,
                geom=self.geom,
                visible_geom=visible_geom,
                visible_text=visible_text,
                )
        
        ani = animation.FuncAnimation(fig, update, np.arange(len(joint_path)), interval=interval, repeat=repeat)
        self.show()

    def show(self):
        self.render.show()

    def reset(self):
        self.obj_collision_mngr = None
        self.scene.objs = OrderedDict()
        
        if self.scene.robot is not None:
            self.robot_collision_mngr = None            
            self.scene.robot = None
        if self.scene.robot.has_gripper:
            self.gripper_collision_mngr = None
            self.scene.robot.gripper = None

    def copy_scene(self, scene_mngr=None):
        copied_scene = SceneManager()
        if scene_mngr is None:
            scene_mngr = self
        for k,v in scene_mngr.__dict__.items():
            if not "collision_mngr" in k:
                copied_scene.__dict__[k] = deepcopy(v)
            else:
                copied_scene.__dict__[k] = v
        return copied_scene

    @property
    def gripper_name(self):
        if not self.scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
        
        return self.scene.robot.gripper.name