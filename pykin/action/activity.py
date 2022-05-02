import numpy as np
from abc import abstractclassmethod, ABCMeta
from dataclasses import dataclass

import pykin.utils.plot_utils as plt
from pykin.scene.scene import SceneManager
from pykin.utils.action_utils import surface_sampling


@dataclass
class ActionInfo:
    ACTION = "action"
    PICK_OBJ_NAME = "pick_obj_name"
    HELD_OBJ_NAME = "held_obj_name"
    PLACE_OBJ_NAME = "place_obj_name"
    GRASP_POSES = "grasp_poses"
    PRE_GRASP_POSES = "pre_grasp_poses"
    POST_GRASP_POSES = "post_grasp_poses"
    TCP_POSES = "tcp_poses"
    RELEASE_POSES = "release_poses"
    LEVEL = "level"
    
class ActivityBase(metaclass=ABCMeta):
    """
    Activity Base class

    Args:
        robot (SingleArm or Bimanual): manipulator type
        robot_col_mngr (CollisionManager): robot's CollisionManager
        object_mngr (ObjectManager): object's Manager
    """
    def __init__(
        self,
        scene_mngr:SceneManager
    ):
        self.scene_mngr = scene_mngr
        self.action_info = ActionInfo

    def __repr__(self) -> str:
        return 'pykin.action.activity.{}()'.format(type(self).__name__)

    @abstractclassmethod
    def get_possible_actions_level_1(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_possible_ik_solve_level_2(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_possible_transitions(self):
        raise NotImplementedError

    def get_surface_points_from_mesh(self, mesh, n_sampling=100, weights=None):
        contact_points, _, normals = surface_sampling(mesh, n_sampling, weights)
        return contact_points, normals

    def _collide(self, is_only_gripper:bool)->bool:
        collide = False
        if is_only_gripper:
            collide = self.scene_mngr.collide_objs_and_gripper()
        else:
            collide = self.scene_mngr.collide_objs_and_robot()
        return collide

    def _solve_ik(self, pose1, pose2, eps=1e-3):
        pose_error = self.scene_mngr.scene.robot.get_pose_error(pose1, pose2)
        if pose_error < eps:
            return True
        return False

    def render_points(self, ax, points, s=5, c='r'):
        if isinstance(points, list):
            points = np.array(points).reshape(-1,3)
        for point in points:
            ax.scatter(point[0], point[1], point[2], s=5, c='r')

    def render_point(self, ax, point, s=5, c='r'):
        ax.scatter(point[0], point[1], point[2], s=5, c='r')

    def render_axis(
        self,
        ax,
        pose,
        axis=[1, 1, 1],
        scale=0.05
    ):
        plt.render_axis(ax, pose, axis, scale)

    def show(self):
        self.scene_mngr.show()