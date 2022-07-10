import numpy as np
import matplotlib.animation as animation
from collections import OrderedDict
from copy import deepcopy

from pykin.scene.scene import Scene
from pykin.scene.object import Object
from pykin.scene.render import RenderPyPlot, RenderTriMesh
from pykin.robots.single_arm import SingleArm
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.mesh_utils import get_relative_transform
import pykin.utils.plot_utils as p_utils

class SceneManager:
    def __init__(
        self, 
        geom="collision", 
        is_pyplot=True, 
        scene:Scene=None,
        benchmark:dict={1 : {'stack_num': 3, 'goal_object':'goal_box'}}
    ):
        # Element for Scene
        self.geom = geom
        self._scene = scene
        if scene is None:
            self._scene = Scene(benchmark)

        self.init_objects = OrderedDict()
        self.init_logical_states = OrderedDict()

        self.attached_obj_name = None
        self.save_grasp_pose = {}
        
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
        if name in self._scene.objs:
            raise ValueError(
                "Duplicate name: object {} already exists".format(name)
            )

        if h_mat is None:
            h_mat = np.eye(4, dtype=np.float32)

        self._scene.objs[name] = Object(name, gtype, gparam, h_mat, color)
        self.obj_collision_mngr.add_object(name, gtype, gparam, h_mat)

        self.init_objects[name] = deepcopy(self._scene.objs[name])

    def add_robot(self, robot:SingleArm, thetas=[]):
        if self._scene.robot is not None:
            raise ValueError(
                "robot {} already exists".format(robot.robot_name)
            )
        self._scene.robot = robot
        
        if np.array(thetas).size != 0:
            self._scene.robot.set_transform(thetas)
        else:
            self._scene.robot.set_transform(robot.init_qpos)

        self.robot_collision_mngr = CollisionManager(is_robot=True)
        self.robot_collision_mngr.setup_robot_collision(robot, geom=self.geom)

        if self._scene.robot.has_gripper:
            self.gripper_collision_mngr = CollisionManager()
            self.gripper_collision_mngr.setup_gripper_collision(robot)
        
    def remove_object(self, name):
        if name not in self._scene.objs:
            raise ValueError("object {} needs to be added first".format(name))
        
        self._scene.objs.pop(name, None)
        self.obj_collision_mngr.remove_object(name)

    def attach_object_on_gripper(self, name, has_transform_bet_gripper_n_obj=False):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")
        
        if name not in self._scene.objs:
            raise ValueError("object {} needs to be added first".format(name))

        self.is_attached = True
        self._scene.robot.gripper.is_attached = self.is_attached
        self.attached_obj_name = self._scene.objs[name].name
        self._scene.robot.gripper.attached_obj_name = self._scene.objs[name].name
        
        self.obj_collision_mngr.remove_object(name)
        
        if has_transform_bet_gripper_n_obj:
            self._transform_bet_gripper_n_obj = self._scene.robot.gripper.transform_bet_gripper_n_obj
        else:
            eef_pose = self.get_gripper_pose()
            self._transform_bet_gripper_n_obj = get_relative_transform(eef_pose, self._scene.objs[name].h_mat)

        self.robot_collision_mngr.add_object(
            self._scene.objs[name].name,
            self._scene.objs[name].gtype,
            self._scene.objs[name].gparam,
            self._scene.objs[name].h_mat)
        self._scene.robot.info["collision"][name] = [self._scene.objs[name].name, self._scene.objs[name].gtype, self._scene.objs[name].gparam, self._scene.objs[name].h_mat]
        self._scene.robot.info["visual"][name] = [self._scene.objs[name].name, self._scene.objs[name].gtype, self._scene.objs[name].gparam, self._scene.objs[name].h_mat]

        self.gripper_collision_mngr.add_object(
            self._scene.objs[name].name,
            self._scene.objs[name].gtype,
            self._scene.objs[name].gparam,
            self._scene.objs[name].h_mat)       
        
        self._scene.robot.gripper.info[name] = [self._scene.objs[name].name, self._scene.objs[name].gtype, self._scene.objs[name].gparam, self._scene.objs[name].h_mat, self._scene.objs[name].color]
        self._scene.objs.pop(name, None)

    def detach_object_from_gripper(self, attached_object=None):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")

        if attached_object is None:
            attached_object = self.attached_obj_name

        self.robot_collision_mngr.remove_object(attached_object)
        self._scene.robot.info["collision"].pop(attached_object)
        self._scene.robot.info["visual"].pop(attached_object)

        self.gripper_collision_mngr.remove_object(attached_object)
        self._scene.robot.gripper.info.pop(attached_object)

        self.is_attached = False
        self._scene.robot.gripper.is_attached = False
        # self._scene.robot.gripper.attached_obj_name = None

    def set_logical_state(self, obj_name, state:tuple):
        if isinstance(state[1], str):
            self._scene.logical_states[obj_name] = {state[0] : self._scene.objs[state[1]]}
        else:
            self._scene.logical_states[obj_name] = {state[0] : state[1]}

    def get_object_pose(self, name):
        if name not in self._scene.objs:
            raise ValueError("object {} needs to be added first".format(name))

        return self._scene.objs[name].h_mat

    def set_object_pose(self, name, pose):
        if name not in self._scene.objs:
            raise ValueError("object {} needs to be added first".format(name))

        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        if "hanoi_disk" in name:
            test = '_'.join(self._scene.objs[name].name.split('_')[:-1])
            for j in range(7):
                disk_name = test + "_" + str(j)
                self._scene.objs[disk_name].h_mat = pose
                self.obj_collision_mngr.set_transform(disk_name, pose)
        else:
            self._scene.objs[name].h_mat = pose
            self.obj_collision_mngr.set_transform(name, pose)

    def compute_ik(self, pose=np.eye(4), method="LM", max_iter=100):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")

        pose = np.asarray(pose)
        if pose.shape != (4,4):
            raise ValueError("Expecting the shape of the pose to be (4,4), instead got: "
                             "{}".format(pose.shape))

        return self._scene.robot.inverse_kin(
            current_joints=np.random.randn(self._scene.robot.arm_dof),
            target_pose=pose,
            method=method,
            max_iter=max_iter)

    def get_robot_eef_pose(self):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")

        return self._scene.robot.info[self.geom][self._scene.robot.eef_name][3]

    def set_robot_eef_pose(self, thetas):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")

        self._scene.robot.set_transform(thetas)
        for link, info in self._scene.robot.info[self.geom].items():
            if link in self.robot_collision_mngr._objs:
                self.robot_collision_mngr.set_transform(link, info[3])
            
            if self._scene.robot.has_gripper:
                if link in self.gripper_collision_mngr._objs:
                    self.gripper_collision_mngr.set_transform(link, info[3])

        if self.is_attached:
            attached_obj_pose = np.dot(self.get_gripper_pose(), self._transform_bet_gripper_n_obj)
            self._scene.robot.info["collision"][self.attached_obj_name][3] = attached_obj_pose
            self._scene.robot.info["visual"][self.attached_obj_name][3] = attached_obj_pose
            self._scene.robot.gripper.info[self.attached_obj_name][3] = attached_obj_pose
            self.robot_collision_mngr.set_transform(self.attached_obj_name, attached_obj_pose)

    def get_gripper_pose(self):
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self._scene.robot.gripper.get_gripper_pose()

    def set_gripper_pose(self, pose=np.eye(4)):
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        self._scene.robot.gripper.set_gripper_pose(pose)
        for link, info in self._scene.robot.gripper.info.items():
            if link in self.gripper_collision_mngr._objs:
                self.gripper_collision_mngr.set_transform(link, info[3])

        if self.is_attached:
            attached_obj_pose = np.dot(self.get_gripper_pose(), self._transform_bet_gripper_n_obj)
            self._scene.robot.gripper.info[self.attached_obj_name][3] = attached_obj_pose
            self.gripper_collision_mngr.set_transform(self.attached_obj_name, attached_obj_pose)

    def get_gripper_tcp_pose(self):
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        return self._scene.robot.gripper.get_gripper_tcp_pose()

    def set_gripper_tcp_pose(self, pose=np.eye(4)):
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        self._scene.robot.gripper.set_gripper_tcp_pose(pose)
        for link, info in self._scene.robot.gripper.info.items():
            if link in self.gripper_collision_mngr._objs:
                self.gripper_collision_mngr.set_transform(link, info[3])

    def collide_objs_and_robot(self, return_names=False):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")
        return self.robot_collision_mngr.in_collision_other(self.obj_collision_mngr, return_names)

    def collide_self_robot(self, return_names=False):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")
        return self.robot_collision_mngr.in_collision_internal(return_names)

    def collide_objs_and_gripper(self, return_names=False):
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
    
        return self.gripper_collision_mngr.in_collision_other(self.obj_collision_mngr, return_names)

    def update_logical_states(self, init=False):
        self._scene.update_logical_states()
        if init:
            self.init_logical_states = self._scene.logical_states

    def get_objs_info(self):
        return self._scene.objs

    def get_robot_info(self):
        if self._scene.robot is None:
            raise ValueError("Robot needs to be added first")
        
        return self._scene.robot.info

    def get_gripper_info(self):
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
        
        return self._scene.robot.gripper.info

    def show_scene_info(self):
        self._scene.show_scene_info()

    def show_logical_states(self):
        self._scene.show_logical_states()

    def render_debug(self, title="Error Scene"):
        fig, ax = p_utils.init_3d_figure(name=title)
        self.render_scene(ax)
        if self.scene.grasp_poses:
            self.render.render_axis(ax, self.scene.grasp_poses["grasp"])
            self.render.render_axis(ax, self.scene.grasp_poses["pre_grasp"])
            self.render.render_axis(ax, self.scene.grasp_poses["post_grasp"])
        if self.scene.release_poses:
            self.render.render_axis(ax, self.scene.release_poses["release"])
            self.render.render_axis(ax, self.scene.release_poses["pre_release"])
            self.render.render_axis(ax, self.scene.release_poses["post_release"])
        self.show()

    def render_scene(
        self, 
        ax=None,
        scene=None,
        alpha=0.9, 
        robot_color=None,
        only_visible_geom=True,
        visible_text=False
    ):
        scene = scene
        if scene is None:
            scene = self._scene
            
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
                only_visible_geom=only_visible_geom,
                visible_text=visible_text)
        else:
            self.render = RenderTriMesh()
            self.render.render_scene(objs=scene.objs, robot=scene.robot, geom=self.geom)
            
    def render_objects_and_gripper(
        self, 
        ax=None, 
        scene=None,
        alpha=0.8, 
        robot_color=None, 
        visible_tcp=True
    ):
        scene = scene
        if scene is None:
            scene = self._scene

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

    def render_objects(self, ax=None, scene=None, alpha=1.0):
        scene = scene
        if scene is None:
            scene = self._scene

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
        only_visible_geom=True,
        visible_text=False
    ):
        scene = scene
        if scene is None:
            scene = self._scene

        if scene.robot is None:
            raise ValueError("Robot needs to be added first")

        if self.is_pyplot:
            self.render.render_robot(
                ax, scene.robot, alpha, robot_color, self.geom, only_visible_geom, visible_text)
        else:
            self.render = RenderTriMesh()
            self.render.render_robot(scene.robot, self.geom)

    def render_gripper(
        self, 
        ax=None, 
        scene=None,
        alpha=1.0, 
        robot_color=None, 
        visible_tcp=True, 
        pose=None,
        only_visible_axis=False,
    ):
        scene = scene
        if scene is None:
            scene = self._scene

        if not scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")

        if self.is_pyplot:
            self.render.render_gripper(
                ax=ax, 
                robot=scene.robot, 
                alpha=alpha, 
                robot_color=robot_color, 
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
        init_scene=None,
        alpha=0.3, 
        robot_color=None,
        joint_path=[], 
        eef_poses=[], 
        visible_gripper=False,
        visible_text=True,
        interval=50,
        repeat=True,
        pick_object=None,
        attach_idx:list = None,
        detach_idx:list = None,
        place_obj_pose=None
    ):
        if not self.is_pyplot:
            ValueError("Only pyplot can render.")
        
        if init_scene is not None:
            self._scene = deepcopy(init_scene)
        
        if pick_object is None:
            pick_object = self.attached_obj_name

        def update(i):
            ax.clear()
            ax._axis3don = False

            if self._scene.objs:
                self.render.render_objects(ax, self._scene.objs, alpha)
            
            if eef_poses is not None:
                self.render.render_trajectory(ax, eef_poses, size=0.1)
            
            self.set_robot_eef_pose(joint_path[i])

            if attach_idx is not None:
                if i in attach_idx:
                    idx = attach_idx.index(i)
                    self.attach_object_on_gripper(pick_object[idx], False)
            
            if detach_idx is not None:
                if i in detach_idx:
                    idx = detach_idx.index(i)
                    if place_obj_pose is None:
                        object_pose = self.get_gripper_info()[pick_object[idx]][3]
                    else:
                        object_pose = place_obj_pose[idx]
                    self.detach_object_from_gripper(pick_object[idx])
                    self.add_object(name=pick_object[idx],
                                    gtype=self.init_objects[pick_object[idx]].gtype,
                                    gparam=self.init_objects[pick_object[idx]].gparam,
                                    h_mat=object_pose,
                                    color=self.init_objects[pick_object[idx]].color)

            visible_geom = True
            if visible_gripper:
                visible_geom = False
                
            self.render.render_robot(
                ax=ax,
                robot=self._scene.robot,
                alpha=alpha,
                robot_color=robot_color,
                geom=self.geom,
                only_visible_geom=visible_geom,
                visible_text=visible_text,
                visible_gripper=visible_gripper,
            )
            
            if i == len(joint_path)-1:
                print("Animation Finished..")
        ani = animation.FuncAnimation(fig, update, np.arange(len(joint_path)), interval=interval, repeat=repeat)
        self.show()

    def show(self):
        self.render.show()

    def reset(self):
        self.obj_collision_mngr = None
        self._scene.objs = OrderedDict()
        
        if self._scene.robot is not None:
            self.robot_collision_mngr = None            
            self._scene.robot = None
        if self._scene.robot.has_gripper:
            self.gripper_collision_mngr = None
            self._scene.robot.gripper = None

    def deepcopy_scene(self, scene_mngr=None):
        copied_scene = SceneManager(benchmark=self.scene.benchmark_config)
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
        if not self._scene.robot.has_gripper:
            raise ValueError("Robot doesn't have a gripper")
        
        return self._scene.robot.gripper.name

    @property
    def scene(self):
        return self._scene

    @scene.setter
    def scene(self, scene):
        self._scene = scene