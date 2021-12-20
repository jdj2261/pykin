import numpy as np
import trimesh
import copy
np.seterr(divide='ignore', invalid='ignore')
import pykin.utils.plot_utils as plt
import pykin.utils.transform_utils as t_utils
from pykin.utils.log_utils import create_logger

logger = create_logger('Grasping Manager', "debug")

class GraspManager:
    def __init__(
        self, 
        gripper=None, 
        max_width=None,
        self_c_manager=None,
        obstacle_c_manager=None
        ):
        if gripper is not None:
            self.gripper = gripper

        self.max_width = max_width
        self.c_manager = self_c_manager
        self.o_manager = obstacle_c_manager
    
        self.mesh_point = np.zeros(3)
        self.contact_points = None
        self.x = None
        self.y = None
        self.z = None
        self.is_joint_limit = False
        self.error_pose = None

    def compute_grasp_pose(self, mesh, obs_pose, approach_distance=0.08, limit_angle=0.02, n_trials=5):
        mesh = copy.deepcopy(mesh)
        mesh.apply_translation(obs_pose)
        while True:
            vertices, normals = self.surface_sampling(mesh, n_samples=2)
            if self.is_force_closure(vertices, normals, limit_angle):
                break
        
        self.contact_points = vertices

        p1 = self.contact_points[0]
        p2 = self.contact_points[1]

        center_point = (p1 + p2) / 2
        line = p2 - p1 

        for i, normal_vector in enumerate(self.compute_normal_vector(line, n_trials)):

            self.y = self.normalize(line)
            
            locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[center_point,],
            ray_directions=[-normal_vector])
            if len(locations) != 0:
                self.mesh_point = locations[0]
            print(i+1)
            self.z = self.normalize(center_point - self.mesh_point)
            self.x = self.normalize(np.cross(self.y, self.z))
        
            grasp_pose = np.eye(4)
            grasp_pose[:3,0] = self.x
            grasp_pose[:3,1] = self.y
            grasp_pose[:3,2] = self.z
            grasp_pose[:3,3] = self.mesh_point - approach_distance * self.z

            yield grasp_pose

    def compute_robust_force_closure(self, mesh, vertices, normals, limit_radian=0.02, n_trials=5):
        sigma = 1e-3
        noise = np.random.normal(0, sigma, (n_trials, 2, 3))
        
        count = 0
        for i in range(n_trials):
            # vertices_copy = vertices.copy()
            new_vertices = vertices + noise[i]

            points, _, faces = trimesh.proximity.closest_point(mesh, new_vertices)        
            normals = mesh.face_normals[faces]

            is_fc = self.is_force_closure(points, normals, limit_radian)
            if is_fc:
                count += 1
        return count/n_trials

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

    def get_pre_grasp_posture(
        self, 
        robot, 
        grasp_pose=None, 
        desired_distance=0.1,
        n_steps=1, 
        epsilon=1e-2
    ):
        assert grasp_pose is not None
        
        pre_grasp_posture = np.eye(4)
        grasp_posure = copy.deepcopy(grasp_pose)

        pre_grasp_posture[:3, :3] = grasp_posure[:3, :3]
        pre_grasp_posture[:3, 3] = grasp_posure[:3, 3] - desired_distance * grasp_posure[:3,2]
        
        transforms, is_get_posture = self._compute_posture(robot, pre_grasp_posture, n_steps, epsilon)
        
        if is_get_posture:
            # self._show_logger(is_get_posture, text="pre")
            if self.collision_free(robot, transforms):
                return transforms, is_get_posture
            print("Pre Collision..")
        self._show_logger(is_get_posture, text="pre")
        return None, False


    def get_grasp_posture(self, robot, grasp_pose=None, n_steps=1, epsilon=1e-5):
        assert grasp_pose is not None
        transforms, is_get_posture = self._compute_posture(robot, grasp_pose, n_steps, epsilon)

        if is_get_posture:
            # self._show_logger(is_get_posture, text="post")
            if self.collision_free(robot, transforms):
                return transforms, is_get_posture
            print("Post Collision..")
        self._show_logger(is_get_posture, text="pre")
        return None, False

    def _compute_posture(self, robot, grasp_pose, n_steps, epsilon):
        eef_pose, qpos, transforms = self._compute_kinematics(robot, grasp_pose)
        is_grasp_success = False

        for _ in range(n_steps):
            transforms = robot.forward_kin(np.array(qpos))
            goal_pose = transforms[robot.eef_name].h_mat

            self.is_joint_limit = robot.check_limit_joint(qpos)
            self.error_pose = robot.get_pose_error(grasp_pose, goal_pose)

            if self.error_pose < epsilon:
                is_grasp_success = True
                break
            qpos = robot.inverse_kin(np.random.randn(len(qpos)), eef_pose, method="LM", maxIter=500)

        if is_grasp_success:
            return transforms, is_grasp_success

        return None, is_grasp_success

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

    def _compute_kinematics(self, robot, grasp_pose):
        eef_pose = t_utils.get_pose_from_homogeneous(grasp_pose)
        qpos = robot.inverse_kin(np.random.randn(7), eef_pose, maxIter=500)
        transforms = robot.forward_kin(np.array(qpos))

        return eef_pose, qpos, transforms

    def _show_logger(self, is_success, text):
        if is_success:
            if text == "pre":
                logger.info(f"Success to get pre grasp posure.")
            else:
                logger.info(f"Success to get grasp posure.")
        else:
            if text == "pre":
                logger.error("Failed to get pre grasp posure.")
            else:
                logger.error("Failed to get grasp posure.")

            logger.error("The pose error is {:.6f}".format(self.error_pose))
            # if not self.is_joint_limit:
            #     logger.error("The joint limit was exceeded.")


    def find_grasp_vertices(self, mesh, vectorA, vectorB):
        vectorAB = vectorB - vectorA
        locations, index_ray, face_ind = mesh.ray.intersects_location(
            ray_origins=[vectorA, vectorB],
            ray_directions=[vectorB - vectorA, vectorA - vectorB],
            multiple_hits=False)
        return locations, face_ind

    def compute_normal_vector(self, vector, n_trials=10):
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
    def surface_sampling(mesh, n_samples=2):
        vertices, face_ind = trimesh.sample.sample_surface(mesh, count=n_samples)
        normals = mesh.face_normals[face_ind]
        return vertices, normals

    @staticmethod
    def find_intersections(mesh, p1, p2):
        ray_origin = (p1 + p2) / 2
        ray_length = np.linalg.norm(p1 - p2)
        ray_dir = (p2 - p1) / ray_length
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[ray_origin, ray_origin],
            ray_directions=[ray_dir, -ray_dir],
            multiple_hits=True)
        if len(locations) == 0:
            return [], []
        dist_to_center = np.linalg.norm(locations - ray_origin, axis=1)
        dist_mask = dist_to_center <= (ray_length / 2) # only keep intersections on the segment.
        on_segment = locations[dist_mask]
        faces = index_tri[dist_mask]
        return on_segment, faces

    @staticmethod
    def projection(v, u):
        return np.dot(v, u) / np.dot(u, u) * u

    @staticmethod
    def normalize(vec):
        return vec / np.linalg.norm(vec)

    def visualize_grasp_pose(self, ax):
        plt.plot_vertices(ax, self.contact_points, s=10, c='red')
        # plt.plot_vertices(ax, self.mesh_point)   
        # plt.plot_normal_vector(ax, self.mesh_point, self.x, scale=0.1, edgecolor="red")    
        # plt.plot_normal_vector(ax, self.mesh_point, self.y, scale=0.1, edgecolor="green")    
        # plt.plot_normal_vector(ax, self.mesh_point, self.z, scale=0.1, edgecolor="blue")  