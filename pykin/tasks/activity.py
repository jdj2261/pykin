import numpy as np
import trimesh
from abc import abstractclassmethod

# import pykin.utils.pnp_utils

import pykin.utils.plot_utils as plt
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.task_utils import get_transform


class ActivityBase:
    def __init__(
        self,
        robot,
        robot_col_manager,
        obstacles_col_manager,
        mesh_path,
        **gripper_configures
    ):
        self.robot = robot
        self.robot_c_manager = robot_col_manager
        self.obstacles_c_manager = obstacles_col_manager
        self.gripper_c_manager = CollisionManager()
        self.mesh_path = mesh_path
        self.gripper_names = gripper_configures.get("gripper_names", None)
        self.gripper_names.insert(0, self.robot.eef_name)
        self.gripper_max_width = gripper_configures.get("gripper_max_width", 0.0)
        self.gripper_max_depth = gripper_configures.get("gripper_max_depth", 0.0)
        self.tcp_position = gripper_configures.get("tcp_position", np.zeros(3))
        self.gripper = self._generate_gripper()

    def __repr__(self) -> str:
        return 'pykin.tasks.activity.{}()'.format(type(self).__name__)

    @abstractclassmethod
    def generate_tcp_poses(self):
        pass

    def _generate_gripper(self):
        gripper = {}
        for link, transform in self.robot.init_transformations.items():
            if link in self.gripper_names:
                gripper[link] = transform.h_mat
                if self.robot.links[link].collision.gtype == "mesh":
                    mesh_name = self.robot.links[link].collision.gparam.get('filename')
                    file_name = self.mesh_path + mesh_name
                    mesh = trimesh.load_mesh(file_name)
                    A2B = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
                    self.gripper_c_manager.add_object(link, gtype="mesh", gparam=mesh, transform=A2B)
        return gripper

    def get_gripper(self):
        return self.gripper

    def get_gripper_transformed(self, pose, is_tcp=True):
        transform_gripper = {}
        gripper = self.get_gripper()
        tcp_pose = pose

        if not is_tcp:
            tcp_pose = self.get_tcp_h_mat_from_eef(pose)

        for link, transform in gripper.items():
            T = get_transform(gripper[self.gripper_names[-1]], tcp_pose)
            transform_gripper[link] = np.dot(T, transform)
        return transform_gripper

    def collision_free(self, transformations, only_gripper=False) -> bool:
        for link, transform in transformations.items():
            if only_gripper:
                if self.robot.links[link].collision.gtype == "mesh":
                    A2B = np.dot(transform, self.robot.links[link].collision.offset.h_mat)
                    self.gripper_c_manager.set_transform(link, A2B)
            else:
                for link, transform in transformations.items():
                    if link in self.robot_c_manager._objs:
                        A2B = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
                        self.robot_c_manager.set_transform(name=link, transform=A2B)
          
        if only_gripper:
            is_obstacle_collision = self.gripper_c_manager.in_collision_other(other_manager=self.obstacles_c_manager)
            if is_obstacle_collision:
                return False
            return True
        else:
            is_self_collision = self.robot_c_manager.in_collision_internal()
            is_obstacle_collision = self.robot_c_manager.in_collision_other(other_manager=self.obstacles_c_manager)
            if is_self_collision or is_obstacle_collision:
                return False
            return True

    def get_tcp_pose(self, transformations):
        return transformations["tcp"]

    def get_eef_h_mat_from_tcp(self, tcp_pose):
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = tcp_pose[:3, :3]
        eef_pose[:3, 3] = tcp_pose[:3, 3] - np.dot(self.tcp_position[-1], tcp_pose[:3, 2])
        return eef_pose

    def get_tcp_h_mat_from_eef(self, eef_pose):
        tcp_pose = np.eye(4)
        tcp_pose[:3, :3] = eef_pose[:3, :3]
        tcp_pose[:3, 3] = eef_pose[:3, 3] + np.dot(self.tcp_position[-1], eef_pose[:3, 2])
        return tcp_pose

    def visualize_robot(
        self, 
        ax, 
        transformations, 
        alpha=1.0,
        only_gripper=False
    ):
        plt.plot_basis(self.robot, ax)
        for link, transform in transformations.items():
            if "pedestal" in link:
                continue
            if self.robot.links[link].collision.gtype == "mesh":
                mesh_name = self.mesh_path + self.robot.links[link].collision.gparam.get('filename')
                mesh = trimesh.load_mesh(mesh_name)
                A2B = np.dot(transform.h_mat, self.robot.links[link].collision.offset.h_mat)
                color = self.robot.links[link].collision.gparam.get('color')
                color = np.array([color for color in color.values()]).flatten()
                mesh.visual.face_colors = color
                
                if only_gripper:
                    if link in self.gripper_names:
                        plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=alpha, color=color)
                else:
                    plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=alpha, color=color)

    def visualize_gripper(
        self,
        ax,
        gripper,
        alpha=1.0
    ):
        plt.plot_basis(self.robot, ax)
        for link, transform in gripper.items():
            if self.robot.links[link].collision.gtype == "mesh":
                mesh_name = self.mesh_path + self.robot.links[link].collision.gparam.get('filename')
                mesh = trimesh.load_mesh(mesh_name)
                A2B = np.dot(transform, self.robot.links[link].collision.offset.h_mat)
                color = self.robot.links[link].collision.gparam.get('color')
                color = np.array([color for color in color.values()]).flatten()
                mesh.visual.face_colors = color
                if "finger" in link:
                    alpha = 1
                plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=alpha, color=color)

    def visualize_axis(
        self,
        ax,
        transformation,
        link=None,
        axis=[1, 1, 1],
        scale=0.1
        
    ):
        pose = transformation
        if link is not None:
            pose = transformation[link].h_mat
        plt.plot_basis(self.robot, ax)
        if axis[0]:
            plt.plot_normal_vector(ax, pose[:3, 3], pose[:3, 0], scale=scale, edgecolor="red")
        if axis[1]:
            plt.plot_normal_vector(ax, pose[:3, 3], pose[:3, 1], scale=scale, edgecolor="green")
        if axis[2]:
            plt.plot_normal_vector(ax, pose[:3, 3], pose[:3, 2], scale=scale, edgecolor="blue")

    def visualize_point(
        self,
        ax,
        transformation,
        link=None
    ):
        pose = transformation
        if link is not None:
            pose = transformation[link].h_mat

        plt.plot_basis(self.robot, ax)
        plt.plot_vertices(ax, pose[:3, 3])   