import numpy as np
import trimesh
from abc import abstractclassmethod

import pykin.utils.plot_utils as plt
from pykin.collision.collision_manager import CollisionManager
from pykin.objects.object_manager import ObjectManager
from pykin.objects.gripper import Gripper


class ActivityBase:
    """
    Activity Base class

    Args:
        robot (SingleArm or Bimanual): manipulator type
        robot_col_mngr (CollisionManager): robot's CollisionManager
        object_mngr (ObjectManager): object's Manager
    """
    def __init__(
        self,
        robot,
        robot_col_mngr: CollisionManager,
        object_mngr: ObjectManager,
    ):
        self.robot = robot
        self.robot_col_mngr = robot_col_mngr
        self.object_mngr = object_mngr
        self.mesh_path = self.robot_col_mngr.mesh_path

    def __repr__(self) -> str:
        return 'pykin.tasks.activity.{}()'.format(type(self).__name__)

    @abstractclassmethod
    def generate_tcp_poses(self):
        """
        Generate tcp poses
        """
        pass

    def _collision_free(self, fk, only_gripper=False):
        """
        Check collision free or not

        Args:
            fk (OrderedDict)
            only_gripper (bool): if only gripper or not

        Returns:
            bool: If collision free, then true 
                  Otherwise, then false
        """
        self._set_transform_col_manager(fk, only_gripper)
        if only_gripper:
            is_object_collision = self.gripper_c_manager.in_collision_other(other_manager=self.object_c_manager)
            if is_object_collision:
                return False
            return True
        else:
            is_self_collision = self.robot_col_mngr.in_collision_internal(return_names=False)
            is_object_collision = self.robot_col_mngr.in_collision_other(other_manager=self.object_c_manager, return_names=False)
            if is_self_collision or is_object_collision:
                return False
            return True

    def _set_transform_col_manager(self, fk, only_gripper):
        """
        Set transform collision manager

        Args:
            fk (OrderedDict)
            only_gripper (bool): if only gripper or not
        """
        for link, transform in fk.items():
            if only_gripper:
                if self.object_mngr.gripper_manager.gripper[link][2] == "mesh":
                    self.gripper.col_manager.set_transform(link, transform)
            else:
                if link in self.robot_col_mngr._objs:
                    if self.robot_col_mngr.geom == "visual":
                        h_mat = np.dot(transform.h_mat, self.robot.links[link].visual.offset.h_mat)
                        self.robot_col_mngr.set_transform(name=link, h_mat=h_mat)
                    else:
                        h_mat = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
                        self.robot_col_mngr.set_transform(name=link, h_mat=h_mat)
        
    def _check_ik_solution(self, eef_pose, goal_pose, eps=1e-2):
        """
        Check ik solution's error

        Args:
            eef_pose (np.array): eef pose
            goal_pose (np.array): goal pose
            eps (float): Threshold of pose error

        Returns:
            bool: If the pose error is less than eps, then true
                  Otherwise, then false
        """
        error_pose = self.robot.get_pose_error(eef_pose, goal_pose)
        if error_pose < eps:
            return True
        return False

    def visualize_robot(
        self, 
        ax, 
        fk, 
        alpha=1.0,
        only_gripper=False
    ):
        """
        Visualize robot

        Args:
            ax (Axes3DSubplot)
            fk (np.array)
            alpha (float): transparency(투명도)
            only_gripper (bool): If True, then visualize only gripper
                                 Otherwise, visualize robot
        """
        plt.plot_basis(ax, self.robot)
        for link, transform in fk.items():
            if "pedestal" in link:
                continue
            if self.robot.links[link].collision.gtype == "mesh":
                
                mesh_name = self.mesh_path + self.robot.links[link].collision.gparam.get('filename')
                mesh = trimesh.load_mesh(mesh_name)
                h_mat = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
                color = self.robot.links[link].collision.gparam.get('color')
                color = np.array([color for color in color.values()]).flatten()
                mesh.visual.face_colors = color
                
                if only_gripper:
                    if link in self.gripper_names:
                        plt.plot_mesh(ax=ax, mesh=mesh, h_mat=h_mat, alpha=alpha, color=color)
                else:
                    plt.plot_mesh(ax=ax, mesh=mesh, h_mat=h_mat, alpha=alpha, color=color)

    def visualize_axis(
        self,
        ax,
        h_mat,
        axis=[1, 1, 1],
        scale=0.1,
        visible_basis=False
    ):
        """
        Visualize axis

        Args:
            ax (Axes3DSubplot)
            h_mat (np.array)
            axis (np.array): As for the axis(x, y, z) you want to see, 1 or 0.
            scale (float): scale of axis
            visible_basis (bool): If visible basis, then plot basis from robot base pose
        """
        if visible_basis:
            plt.plot_basis(ax, self.robot)
        pose = h_mat
        if axis[0]:
            plt.plot_normal_vector(ax, pose[:3, 3], pose[:3, 0], scale=scale, edgecolor="red")
        if axis[1]:
            plt.plot_normal_vector(ax, pose[:3, 3], pose[:3, 1], scale=scale, edgecolor="green")
        if axis[2]:
            plt.plot_normal_vector(ax, pose[:3, 3], pose[:3, 2], scale=scale, edgecolor="blue")