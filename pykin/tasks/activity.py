import numpy as np
import trimesh
from abc import abstractclassmethod

import pykin.utils.plot_utils as plt
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.task_utils import get_absolute_transform


class ActivityBase:
    """
    Activity Base class

    Args:
        robot (SingleArm or Bimanual): manipulator type
        robot_col_manager (CollisionManager): robot's CollisionManager
        object_col_manager (CollisionManager): object's CollisionManager
        mesh_path (str): absolute path of mesh
        gripper_configures (dict): configurations of gripper
                                   (gripper's name, max width, max depth, tcp position)
    """
    def __init__(
        self,
        robot,
        robot_col_manager,
        objects_col_manager,
        mesh_path,
        **gripper_configures
    ):
        self.robot = robot
        self.robot_c_manager = robot_col_manager
        self.object_c_manager = objects_col_manager
        self.gripper_c_manager = CollisionManager()
        self.mesh_path = mesh_path

        self.gripper_names = gripper_configures.get("gripper_names", None)
        self.gripper_names.insert(0, self.robot.eef_name)
        self.gripper_max_width = gripper_configures.get("gripper_max_width", 0.0)
        self.gripper_max_depth = gripper_configures.get("gripper_max_depth", 0.0)
        self.tcp_position = gripper_configures.get("tcp_position", np.zeros(3))
        self.gripper = self._get_gripper_fk()

    def __repr__(self) -> str:
        return 'pykin.tasks.activity.{}()'.format(type(self).__name__)

    def _get_gripper_fk(self):
        """
        Get only gripper forward kinematics and setup gripper collision manager

        Returns:
            gripper (dict)
        """
        gripper = {}
        for link, transform in self.robot.init_fk.items():
            if link in self.gripper_names:
                gripper[link] = transform.h_mat
                self._setup_gripper_col_manager(link, transform)
        return gripper

    def _setup_gripper_col_manager(self, link, transform):
        """
        Setup Gripper Collision Manager

        Args:
            link (str): link name
            transform (np.array): transformation matrix
        """
        if self.robot.links[link].collision.gtype == "mesh":
            mesh_name = self.robot.links[link].collision.gparam.get('filename')
            file_name = self.mesh_path + mesh_name
            mesh = trimesh.load_mesh(file_name)
            h_mat = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
            self.gripper_c_manager.add_object(link, gtype="mesh", gparam=mesh, h_mat=h_mat)

    @abstractclassmethod
    def generate_tcp_poses(self):
        """
        Generate tcp poses
        """
        pass

    def get_transformed_gripper_fk(self, pose, is_tcp=True):
        """
        Get transformed gripper forward kinematics

        Args:
            pose (np.array): eef pose or tcp pose
            is_tcp (bool): If True, pose is tcp pose
                           Otherwise, pose is eef pose
        
        Returns:
            gripper_transformed (dict)
        """
        transformed_gripper_fk = {}
        gripper_fk = self.get_gripper()
        tcp_pose = pose

        if not is_tcp:
            tcp_pose = self.get_tcp_h_mat_from_eef(pose)

        for link, transform in gripper_fk.items():
            T = get_absolute_transform(gripper_fk[self.gripper_names[-1]], tcp_pose)
            transformed_gripper_fk[link] = np.dot(T, transform)
        return transformed_gripper_fk

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
            is_self_collision = self.robot_c_manager.in_collision_internal(return_names=False)
            is_object_collision = self.robot_c_manager.in_collision_other(other_manager=self.object_c_manager, return_names=False)
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
                if self.robot_c_manager.geom == "visual":
                    if self.robot.links[link].visual.gtype == "mesh":
                        h_mat = np.dot(transform, self.robot.links[link].visual.offset.h_mat)
                        self.gripper_c_manager.set_transform(link, h_mat)
                else:
                    if self.robot.links[link].collision.gtype == "mesh":
                        h_mat = np.dot(transform, self.robot.links[link].collision.offset.h_mat)
                        self.gripper_c_manager.set_transform(link, h_mat)
            else:
                if link in self.robot_c_manager._objs:
                    if self.robot_c_manager.geom == "visual":
                        h_mat = np.dot(transform.h_mat, self.robot.links[link].visual.offset.h_mat)
                        self.robot_c_manager.set_transform(name=link, h_mat=h_mat)
                    else:
                        h_mat = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
                        self.robot_c_manager.set_transform(name=link, h_mat=h_mat)
        
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

    def get_gripper(self):
        """
        Get gripper
        """
        return self.gripper

    def get_eef_h_mat_from_tcp(self, tcp_pose):
        """
        Get eef transformation matrix from tcp pose

        Args:
            tcp_pose (np.array): tcp pose
        
        Returns:
            eef_pose (np.array): eef pose
        """
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = tcp_pose[:3, :3]
        eef_pose[:3, 3] = tcp_pose[:3, 3] - np.dot(self.tcp_position[-1], tcp_pose[:3, 2])
        return eef_pose

    def get_tcp_h_mat_from_eef(self, eef_pose):
        """
        Get tcp transformation matrix from eef pose

        Args:
            eef_pose (np.array): eef pose
        
        Returns:
            tcp_pose (np.array): tcp pose
        """
        tcp_pose = np.eye(4)
        tcp_pose[:3, :3] = eef_pose[:3, :3]
        tcp_pose[:3, 3] = eef_pose[:3, 3] + np.dot(self.tcp_position[-1], eef_pose[:3, 2])
        return tcp_pose

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

    def visualize_gripper(
        self,
        ax,
        fk,
        alpha=1.0,
        color=None,
        visible_basis=False
    ):
        """
        Visualize gripper

        Args:
            ax (Axes3DSubplot)
            fk (np.array)
            alpha (float): transparency(투명도)
            color (str): name of color (blue, red, green etc..)
            visible_basis (bool): If visible basis, then plot basis from robot base pose
        """
        if visible_basis:
            plt.plot_basis(ax, self.robot)
        for link, transform in fk.items():
            if self.robot.links[link].collision.gtype == "mesh":
                mesh_name = self.mesh_path + self.robot.links[link].collision.gparam.get('filename')
                mesh = trimesh.load_mesh(mesh_name)
                h_mat = np.dot(transform, self.robot.links[link].collision.offset.h_mat)

                mesh_color = color
                if color is None:
                    mesh_color = self.robot.links[link].collision.gparam.get('color')
                    mesh_color = np.array([color for color in mesh_color.values()]).flatten()
                if "finger" in link:
                    alpha = 1
                plt.plot_mesh(ax=ax, mesh=mesh, h_mat=h_mat, alpha=alpha, color=mesh_color)

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