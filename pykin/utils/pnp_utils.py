import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import trimesh
import copy

import pykin.utils.plot_utils as plt
import pykin.utils.transform_utils as t_utils
from pykin.utils.log_utils import create_logger

logger = create_logger('Grasping Manager', "debug")

class PnPManager:
    def __init__(
        self, 
        gripper=None, 
        gripper_max_width=None,
        self_c_manager=None,
        obstacle_c_manager=None
        ):
        if gripper is not None:
            self.gripper = gripper

        self.max_width = gripper_max_width
        self.c_manager = self_c_manager
        self.o_manager = obstacle_c_manager
    
        self.mesh_point = np.zeros(3)
        self.contact_points = None
        self.x = None
        self.y = None
        self.z = None
        self.error_pose = None
        self.grasp_transforms = None

    def get_all_grasp_transforms(
        self,
        robot,
        obj_mesh,
        obj_pose,
        approach_distance,
        limit_angle,
        n_trials
    ):
        while True:
            grasp_poses = self.compute_grasp_pose(
                obj_mesh, 
                obj_pose, 
                approach_distance, 
                limit_angle, 
                n_trials)
            for grasp_pose in grasp_poses:
                transforms, is_grasp_success = self.get_transforms(robot, grasp_pose, epsilon=0.1)
                pre_transforms, is_post_grasp_success = self.get_pre_transforms(robot, grasp_pose, desired_distance=0.14, epsilon=0.1)
                if is_grasp_success and is_post_grasp_success:
                    break
            if is_grasp_success and is_post_grasp_success:
                break
        
        self.grasp_transforms = transforms
        return transforms, pre_transforms

    def get_all_release_transforms(
        self, 
        robot,
        obj_mesh, 
        obj_pose,
        approach_distance,
        n_trials=5,
        n_samples=10,
    ):
        while True:
            is_release_success = False
            
            release_poses = self.compute_release_pose(
                robot,
                obj_mesh,
                obj_pose,
                approach_distance,
                n_trials,
                n_samples
            )
            
            for release_pose in release_poses:
                transforms, is_release_success = self.get_transforms(robot, release_pose, epsilon=0.1)
                # TODO 
                # get_pre_transforms
                if is_release_success:
                    break
                logger.warning("Retry release pose")
            if is_release_success :
                break

        return transforms

    def compute_grasp_pose(
        self, 
        obj_mesh, 
        obs_pose, 
        approach_distance=0.08, 
        limit_angle=0.02, 
        n_trials=5
    ):
        self.pick_object_mesh = obj_mesh
        self.pick_object_pose = obs_pose

        mesh = copy.deepcopy(obj_mesh)    
        mesh.apply_transform(obs_pose)
        
        while True:
            vertices, _, normals = self.surface_sampling(mesh, n_samples=2)
            if self.is_force_closure(vertices, normals, limit_angle):
                break
        
        self.contact_points = vertices
        p1 = self.contact_points[0]
        p2 = self.contact_points[1]

        center_point = (p1 + p2) / 2
        line = p2 - p1 

        for i, normal_vector in enumerate(self._compute_normal_directions(line, n_trials)):
            logger.debug("{} Get Grasp pose".format(i+1))
            self.y = self.normalize(line)
            
            locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[center_point,],
            ray_directions=[-normal_vector])
            if len(locations) != 0:
                self.mesh_point = locations[0]
            self.z = self.normalize(center_point - self.mesh_point)
            self.x = self.normalize(np.cross(self.y, self.z))
        
            grasp_pose = np.eye(4)
            grasp_pose[:3,0] = self.x
            grasp_pose[:3,1] = self.y
            grasp_pose[:3,2] = self.z
            grasp_pose[:3,3] = self.mesh_point - approach_distance * self.z

            yield grasp_pose

    # TODO approach_distance
    def compute_release_pose(
        self,
        robot,
        obj_mesh, 
        obs_pose, 
        approach_distance=0.08, 
        n_trials=5,
        n_samples=10
    ):
        if self.grasp_transforms is None:
            raise TypeError("Check grasp_transforms")

        mesh = copy.deepcopy(obj_mesh)
        mesh.apply_transform(obs_pose)

        weights = np.zeros(len(mesh.faces))
        for idx, vertex in enumerate(mesh.vertices[mesh.faces]):
            weights[idx]=0.0
            if np.all(vertex[:,2] >= mesh.bounds[1][2]):                
                weights[idx] = 1.0

        self.place_points, face_ind, _ = self.surface_sampling(mesh, n_samples, face_weight=weights)
        for vertex, idx in zip(self.place_points, face_ind):
            for theta in np.linspace(0, np.pi, n_trials):
                self.place_object_mesh = copy.deepcopy(self.pick_object_mesh)
                gripper_pose = self.grasp_transforms[robot.eef_name].h_mat    
                normal_vector = mesh.face_normals[idx]
                gripper_pose[:3,3][:2] = vertex[:2]
                gripper_pose[:3,3][2] = vertex[2] + approach_distance * normal_vector[2]
                R = t_utils.get_matrix_from_rpy([0, 0, theta])
                gripper_pose[:3,:3] = np.dot(R, gripper_pose[:3,:3])

                T = np.dot(gripper_pose, np.linalg.inv(self.grasp_transforms[robot.eef_name].h_mat))
                object_pose = np.dot(T, self.pick_object_pose)
    
                self.place_object_mesh.apply_transform(object_pose)
                center_point = self.place_object_mesh.center_mass
                locations, _, _ = mesh.ray.intersects_location(
                                        ray_origins=[center_point,],
                                        ray_directions=[-normal_vector])
                if len(locations) != 0:
                    support_index = np.where(locations == np.max(locations, axis=0)[2])
                    self.support_point = locations[support_index[0]]
                    yield gripper_pose
                else:
                    logger.warning("Not found support point")
                    continue
                
    def is_force_closure(self, vertices, normals, limit_angle=0.1):
        vectorA = vertices[0]
        vectorB = vertices[1]

        normalA = -normals[0]
        normalB = -normals[1]

        vectorAB = vectorB - vectorA
        distance = np.linalg.norm(vectorAB)

        unit_vectorAB = self.normalize(vectorAB)
        angle_A2AB = np.arccos(normalA.dot(unit_vectorAB))

        unit_vectorBA = -1 * unit_vectorAB
        angle_B2AB = np.arccos(normalB.dot(unit_vectorBA))

        if distance > self.max_width:
            return False

        if angle_A2AB > limit_angle or angle_B2AB > limit_angle:
            return False
        
        return True

    def get_pre_transforms(
        self, 
        robot, 
        grasp_pose=None, 
        desired_distance=0.1,
        epsilon=1e-2
    ):
        logger.debug("Compute the pre grasp pose")
        assert grasp_pose is not None
        
        pre_grasp_pose = np.eye(4)
        grasp_posure = copy.deepcopy(grasp_pose)

        pre_grasp_pose[:3, :3] = grasp_posure[:3, :3]
        pre_grasp_pose[:3, 3] = grasp_posure[:3, 3] - desired_distance * grasp_posure[:3,2]
        
        transforms, is_get_pose = self._compute_transforms(robot, pre_grasp_pose, epsilon)
        
        if is_get_pose:
            if self.collision_free(robot, transforms):
                logger.info(f"Success to get pre pose.\n")
                return transforms, is_get_pose
            logger.error(f"A collision has occurred in pre pose.")
        return None, False

    def get_transforms(self, robot, grasp_pose=None, epsilon=1e-5):
        logger.debug("Compute the grasp pose")
        assert grasp_pose is not None

        transforms, is_get_pose = self._compute_transforms(robot, grasp_pose, epsilon)

        if is_get_pose:
            if self.collision_free(robot, transforms):
                logger.info(f"Success to get pose.\n")
                return transforms, is_get_pose
            logger.error(f"A collision has occurred in pose.")
        return None, False

    def collision_free(self, robot, transformations):
        for link, transformation in transformations.items():
            if link in self.c_manager._objs:
                transform = transformation.h_mat
                A2B = np.dot(transform, robot.links[link].collision.offset.h_mat)
                self.c_manager.set_transform(name=link, transform=A2B)

        is_self_collision = self.c_manager.in_collision_internal()
        is_obstacle_collision = self.c_manager.in_collision_other(other_manager=self.o_manager)

        if is_self_collision or is_obstacle_collision:
            return False
        return True

    def _compute_transforms(self, robot, grasp_pose, epsilon):
        qpos = self._compute_inverse_kinematics(robot, grasp_pose)
        is_success = False

        transforms = robot.forward_kin(np.array(qpos))
        goal_pose = transforms[robot.eef_name].h_mat
        self.error_pose = robot.get_pose_error(grasp_pose, goal_pose)

        if self.error_pose < epsilon:
            is_success = True
            return transforms, is_success

        return None, False

    def _compute_inverse_kinematics(self, robot, grasp_pose):
        eef_pose = t_utils.get_pose_from_homogeneous(grasp_pose)
        qpos = robot.inverse_kin(np.random.randn(7), eef_pose, maxIter=500)
        return qpos

    def _compute_normal_directions(self, vector, n_trials=10):
        norm_vector = self.normalize(vector)
        e1, e2 = np.eye(3)[:2]
        v1 = e1 - self.projection(e1, norm_vector)
        v1 = self.normalize(v1)
        v2 = e2 - self.projection(e2, norm_vector) - self.projection(e2, v1)
        v2 = self.normalize(v2)

        for theta in np.linspace(-np.pi/4, np.pi/4, n_trials):
            normal_dir = np.cos(theta) * v1 + np.sin(theta) * v2
            yield normal_dir

    @staticmethod
    def surface_sampling(mesh, n_samples=2, face_weight=None):
        vertices, face_ind = trimesh.sample.sample_surface(mesh, count=n_samples, face_weight=face_weight)
        normals = mesh.face_normals[face_ind]
        return vertices, face_ind, normals

    @staticmethod
    def projection(v, u):
        return np.dot(v, u) / np.dot(u, u) * u

    @staticmethod
    def normalize(vec):
        return vec / np.linalg.norm(vec)

    def visualize_grasp_pose(self, ax):
        plt.plot_vertices(ax, self.contact_points, s=10, c='red')

    def visualize_robot(
        self, 
        ax, 
        robot, 
        transformations, 
        mesh_path, 
        gripper_names=["right_gripper", "leftfinger", "rightfinger"], 
        alpha=1.0,
        only_gripper=False
    ):
        plt.plot_basis(robot, ax)
        for link, transform in transformations.items():
            if "pedestal" in link:
                continue
            if robot.links[link].collision.gtype == "mesh":
                mesh_name = mesh_path + robot.links[link].collision.gparam.get('filename')
                mesh = trimesh.load_mesh(mesh_name)
                A2B = np.dot(transform.h_mat, robot.links[link].collision.offset.h_mat)
                color = robot.links[link].collision.gparam.get('color')

                if color is None:
                    color = np.array([0.2, 0, 0])
                else:
                    color = np.array([color for color in color.values()]).flatten()
                    if "link" in link:
                        color = np.array([0.2, 0.2, 0.2])
                mesh.visual.face_colors = color
                
                if only_gripper:
                    if link in gripper_names:
                        plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=alpha, color=color)
                else:
                    plt.plot_mesh(ax=ax, mesh=mesh, A2B=A2B, alpha=alpha, color=color)

    def visualize_axis(
        self,
        ax,
        transformations,
        eef_name,
        pose=None
    ):
        if pose is None:
            pose = transformations[eef_name].h_mat

        gripper_pos = pose[:3, 3]
        gripper_ori_x = pose[:3, 0]
        gripper_ori_y = pose[:3, 1]
        gripper_ori_z = pose[:3, 2]

        plt.plot_vertices(ax, gripper_pos)   
        plt.plot_normal_vector(ax, gripper_pos, gripper_ori_x, scale=0.1, edgecolor="red")
        plt.plot_normal_vector(ax, gripper_pos, gripper_ori_y, scale=0.1, edgecolor="green")
        plt.plot_normal_vector(ax, gripper_pos, gripper_ori_z, scale=0.1, edgecolor="blue")