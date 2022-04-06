import numpy as np
from enum import Enum, auto
from collections import OrderedDict
from copy import deepcopy

from pykin.tasks.activity import ActivityBase
from pykin.utils.task_utils import normalize, surface_sampling, projection, get_rotation_from_vectors, get_relative_transform
from pykin.utils.transform_utils import get_pose_from_homogeneous
from pykin.utils.log_utils import create_logger

logger = create_logger('Grasp', "debug")

class GraspStatus(Enum):
    """
    Grasp Status Enum class
    """
    pre_grasp_pose = auto()
    grasp_pose = auto()
    post_grasp_pose = auto()
    pre_release_pose = auto()
    release_pose = auto()
    post_release_pose = auto()

class GraspManager(ActivityBase):
    """
    Grasp Manager class

    Args:
        robot (SingleArm or Bimanual): manipulator type
        robot_col_manager (CollisionManager): robot's CollisionManager
        object_col_manager (CollisionManager): object's CollisionManager
        mesh_path (str): absolute path of mesh
        retreat_distance (float): retreat distance
        release_distance (float): release distance
        gripper_configures (dict): configurations of gripper
                                   (gripper's name, max width, max depth, tcp position)
    """
    def __init__(
        self,
        robot,
        robot_col_manager,
        objects_col_manager,
        mesh_path,
        retreat_distance = 0.1,
        release_distance = 0.01,
        **gripper_configures
    ):
        super().__init__(
            robot,
            robot_col_manager,
            objects_col_manager,
            mesh_path,
            **gripper_configures)

        self.retreat_distance = retreat_distance
        self.release_distance = release_distance
        self.tcp_pose = np.eye(4)
        self.post_release_pose = np.eye(4)
        self.contact_points = None
        
        self.obj_info = None
        self.has_obj = False

    def get_grasp_waypoints(
        self,
        obj_info=None,
        obj_mesh=None,
        obj_pose=None,
        limit_angle=0.1,
        num_grasp=1,
        n_trials=1,
    ):
        """
        Get grasp waypoints(pre grasp pose, grasp pose, post grasp pose)

        Args:
            obj_info (dict): object info (name, gtype, gparam, transform) 
            obj_mesh (trimesh.base.Trimesh): object mesh
            obj_pose (np.array): object pose
            limit_angle (float): angle to satisfy force closure
            num_grasp (int): number of sampling contact points
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points
        
        Returns:
            waypoints (OrderedDict): pre grasp pose, grasp pose, post grasp pose
        """
        waypoints = OrderedDict()

        if obj_info:
            self.has_obj = True
            self.obj_info = obj_info
            obj_mesh = obj_info["gparam"]
            obj_pose = obj_info["transform"]

        grasp_pose = self.get_grasp_pose(obj_mesh, obj_pose, limit_angle, num_grasp, n_trials)    
        waypoints[GraspStatus.pre_grasp_pose] = self.pre_grasp_pose
        waypoints[GraspStatus.grasp_pose] = grasp_pose
        waypoints[GraspStatus.post_grasp_pose] =self.post_grasp_pose

        return waypoints

    def get_grasp_pose(        
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        num_grasp=1,
        n_trials=1,
    ):
        """
        Get grasp pose

        Args:
            obj_mesh (trimesh.base.Trimesh): object mesh
            obj_pose (np.array): object pose
            limit_angle (float): angle to satisfy force closure
            num_grasp (int): number of sampling contact points
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points
        
        Returns:
            grasp_pose (np.array)
        """
        grasp_poses = self.generate_grasps(obj_mesh, obj_pose, limit_angle, num_grasp, n_trials)
        grasp_pose = self.filter_grasps(grasp_poses)     
        return grasp_pose

    def get_pre_grasp_pose(self, grasp_pose):
        """
        Get pre grasp pose

        Args:
            grasp_pose (np.array): grasp pose

        Returns:
            pre_grasp_pose (np.array)
        """
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = grasp_pose[:3, :3]
        pre_grasp_pose[:3, 3] = grasp_pose[:3, 3] - self.retreat_distance * grasp_pose[:3,2]    
        # pre_grasp_pose[:3, 3] = grasp_pose[:3, 3] + np.array([0, 0, self.retreat_distance])  
        return pre_grasp_pose

    def get_post_grasp_pose(self, grasp_pose):
        """
        Get post grasp pose

        Args:
            grasp_pose (np.array): grasp pose

        Returns:
            post_grasp_pose (np.array)
        """
        post_grasp_pose = np.eye(4)
        post_grasp_pose[:3, :3] = grasp_pose[:3, :3] 
        post_grasp_pose[:3, 3] = grasp_pose[:3, 3] - self.retreat_distance * grasp_pose[:3,2] 
        # post_grasp_pose[:3, 3] = grasp_pose[:3, 3] + np.array([0, 0, self.retreat_distance])  
        return post_grasp_pose

    def generate_grasps(
        self,
        obj_mesh,
        obj_pose,
        limit_angle=0.05, #radian
        num_grasp=1,
        n_trials=1
    ):
        """
        Generate grasp poses

        Args:
            obj_mesh (trimesh.base.Trimesh): object mesh
            obj_pose (np.array): object pose
            limit_angle (float): angle to satisfy force closure
            num_grasp (int): number of sampling contact points
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points
        
        Returns:
            eef_pose, gripper_transformed (tuple): eef pose, gripper
        """
        cnt = 0
        while cnt < num_grasp * n_trials:
            tcp_poses = self.generate_tcp_poses(obj_mesh, obj_pose, limit_angle, n_trials)
            for tcp_pose, contact_points, _ in tcp_poses:
                gripper_transformed = self.get_transformed_gripper_fk(tcp_pose)

                if self._collision_free(gripper_transformed, only_gripper=True):
                    self.tcp_pose = tcp_pose
                    self.contact_points = contact_points
                    eef_pose = self.get_eef_h_mat_from_tcp(tcp_pose)
                    cnt += 1
                    yield (eef_pose, gripper_transformed)
    
    def filter_grasps(self, grasp_poses):
        """
        Filter grasp pose

        Args:
            grasp_poses (tuple): eef pose, gripper transformed
        
        Returns:
            grasp_pose (np.array): eef pose
        """
        is_success_filtered = False
        for grasp_pose, _ in grasp_poses:
            qpos = self._compute_inverse_kinematics(grasp_pose)
            grasp_transforms = self.robot.forward_kin(np.array(qpos))
            goal_eef_pose = grasp_transforms[self.robot.eef_name].h_mat
 
            if self._check_ik_solution(grasp_pose, goal_eef_pose) and self._collision_free(grasp_transforms):
                pre_grasp_pose = self.get_pre_grasp_pose(grasp_pose)
                pre_transforms, pre_goal_pose = self._get_goal_pose(pre_grasp_pose)
        
                if self._check_ik_solution(pre_grasp_pose, pre_goal_pose) and self._collision_free(pre_transforms):
                    self.pre_grasp_pose = pre_grasp_pose
                    
                    post_grasp_pose = self.get_post_grasp_pose(grasp_pose)
                    post_transforms, post_goal_pose = self._get_goal_pose(post_grasp_pose)

                    if self.has_obj:
                        self.obj_pre_grasp_pose = self.obj_info["transform"]
                        self.obj_grasp_pose = self.obj_info["transform"]

                        self.T_between_gripper_and_obj = get_relative_transform(grasp_pose, self.obj_info["transform"])
                        obj_post_grasp_pose = np.dot(post_grasp_pose, self.T_between_gripper_and_obj)
                        self.obj_post_grasp_pose = obj_post_grasp_pose
                        self._attach_gripper2object(obj_post_grasp_pose)

                    if self._check_ik_solution(post_grasp_pose, post_goal_pose) and self._collision_free(post_transforms):
                        self.post_grasp_pose = post_grasp_pose
                        is_success_filtered = True
                        break

        if not is_success_filtered:
            logger.error(f"Failed to filter Grasp poses")
            return None

        logger.info(f"Success to get Grasp pose.\n")
        return grasp_pose

    def _attach_gripper2object(self, obj_post_grasp_pose):
        """
        Attach object collision on robot collision

        Args:
            obj_post_grasp_pose (np.array): pose grasp pose of object
        """
        self.robot_c_manager.add_object(
            self.obj_info["name"], 
            gtype=self.obj_info["gtype"], gparam=self.obj_info["gparam"], h_mat=obj_post_grasp_pose)

    def generate_tcp_poses(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        n_trials
    ):
        """
        Generate grasp poses

        Args:
            obj_mesh (trimesh.base.Trimesh): object mesh
            obj_pose (np.array): object pose
            limit_angle (float): angle to satisfy force closure
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points
        
        Returns:
            tcp_pose, contact_points, normals (tuple)
        """
        contact_points, normals = self._generate_contact_points(obj_mesh, obj_pose, limit_angle)
        p1, p2 = contact_points
        center_point = (p1 + p2) /2
        line = p2 - p1

        for _, grasp_dir in enumerate(self._generate_grasp_directions(line, n_trials)):
            y = normalize(line)
            z = grasp_dir
            x = np.cross(y, z)

            tcp_pose = np.eye(4)
            tcp_pose[:3,0] = x
            tcp_pose[:3,1] = y
            tcp_pose[:3,2] = z
            tcp_pose[:3,3] = center_point

            yield (tcp_pose, contact_points, normals)

    def _generate_contact_points(
        self,
        obj_mesh,
        obj_pose,
        limit_angle
    ):
        """
        Generate contact points

        Args:
            obj_mesh (trimesh.base.Trimesh): object mesh
            obj_pose (np.array): object pose
            limit_angle (float): angle to satisfy force closure            
        
        Returns:
            contact_points, normals (tuple)
        """
        copied_mesh = deepcopy(obj_mesh)
        copied_mesh.apply_transform(obj_pose)

        while True:
            contact_points, _, normals = surface_sampling(copied_mesh, n_samples=2)
            if self._is_force_closure(contact_points, normals, limit_angle):
                break
        return (contact_points, normals)

    def _generate_grasp_directions(self, line, n_trials):
        """
        Generate grasp dicrections

        Args:
            line (np.array): line from vectorA to vector B
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points

        Returns:
            normal_dir (float): grasp direction
        """
        norm_vector = normalize(line)
        e1, e2 = np.eye(3)[:2]
        v1 = e1 - projection(e1, norm_vector)
        v1 = normalize(v1)
        v2 = e2 - projection(e2, norm_vector) - projection(e2, v1)
        v2 = normalize(v2)

        for theta in np.linspace(-np.pi, np.pi, n_trials):
            normal_dir = np.cos(theta) * v1 + np.sin(theta) * v2
            yield normal_dir

    def _is_force_closure(self, vertices, normals, limit_angle):
        """
        Check force closure

        Args:
            vertices (np.array): contact points
            normals (np.array): normal vector of contact points
            limit_angle (float): angle to satisfy force closure 

        Returns:
            bool: If satisfy force closure, then true
                  Otherwise, then false
        """
        vectorA = vertices[0]
        vectorB = vertices[1]

        normalA = -normals[0]
        normalB = -normals[1]

        vectorAB = vectorB - vectorA
        distance = np.linalg.norm(vectorAB)

        unit_vectorAB = normalize(vectorAB)
        angle_A2AB = np.arccos(normalA.dot(unit_vectorAB))

        unit_vectorBA = -1 * unit_vectorAB
        angle_B2AB = np.arccos(normalB.dot(unit_vectorBA))

        if distance > self.gripper_max_width:
            return False

        if angle_A2AB > limit_angle or angle_B2AB > limit_angle:
            return False
        
        return True

    def get_release_waypoints(
        self,
        obj_info_on_sup=None,
        n_samples_on_sup=1,
        obj_info_for_sup=None,
        n_samples_for_sup=1,
        n_trials=1,
        obj_mesh_on_sup=None,
        obj_pose_on_sup=None,
        obj_mesh_for_sup=None,
        obj_pose_for_sup=None,
    ):
        """
        Get release waypoints(pre release pose, release pose, post release pose)

        Args:
            obj_info_on_sup (dict): info of support object(name, gtype, gparam, transform) 
            n_samples_on_sup (int): number of sampling points on support object
            obj_info_for_sup (dict): info of grasp object(name, gtype, gparam, transform) 
            n_samples_for_sup (int): number of sampling points on grasp object
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points
            obj_mesh_on_sup (trimesh.base.Trimesh): mesh of support object
            obj_pose_on_sup (np.array): pose of support object
            obj_mesh_for_sup (trimesh.base.Trimesh): mesh of grasp object
            obj_pose_for_sup (np.array): pose of grasp object
        
        Returns:
            waypoints (OrderedDict): pre release pose, release pose, post release pose
        """
        waypoints = OrderedDict()

        if obj_info_on_sup:
            obj_mesh_on_sup = obj_info_on_sup["gparam"]
            obj_pose_on_sup = obj_info_on_sup["transform"]

        if obj_info_for_sup:
            obj_mesh_for_sup = obj_info_for_sup["gparam"]
            obj_pose_for_sup = obj_info_for_sup["transform"]

        release_pose = self.get_release_pose(
            obj_mesh_on_sup,
            obj_pose_on_sup,
            n_samples_on_sup,
            obj_mesh_for_sup,
            obj_pose_for_sup,
            n_samples_for_sup, 
            n_trials)
        
        waypoints[GraspStatus.pre_release_pose] = self.pre_release_pose
        waypoints[GraspStatus.release_pose] = release_pose
        waypoints[GraspStatus.post_release_pose] =self.post_release_pose

        return waypoints
        
    def get_release_pose(
        self,
        obj_mesh_on_sup,
        obj_pose_on_sup,
        n_samples_on_sup,
        obj_mesh_for_sup,
        obj_pose_for_sup,
        n_samples_for_sup,
        n_trials
    ):
        """
        Get release pose

        Args:
            obj_mesh_on_sup (trimesh.base.Trimesh): mesh of support object
            obj_pose_on_sup (np.array): pose of support object
            n_samples_on_sup (int): number of sampling points on support object
            obj_mesh_for_sup (trimesh.base.Trimesh): mesh of grasp object
            obj_pose_for_sup (np.array): pose of grasp object            
            n_samples_for_sup (int): number of sampling points on grasp object
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points        
        
        Returns:
            release_pose (np.array)
        """
        support_poses = self.generate_supports(
            obj_mesh_on_sup,
            obj_pose_on_sup,
            n_samples_on_sup,
            obj_mesh_for_sup,
            obj_pose_for_sup,
            n_samples_for_sup,
            n_trials)

        release_pose = self.filter_supports(support_poses)
        return release_pose

    def get_pre_release_pose(self, release_pose):
        """
        Get pre release pose

        Args:
            release_pose (np.array)
        
        Returns:
            pre_release_pose (np.array)
        """
        pre_release_pose = np.eye(4)
        pre_release_pose[:3, :3] = release_pose[:3, :3]
        pre_release_pose[:3, 3] = release_pose[:3, 3] + np.array([0, 0, self.retreat_distance])
        return pre_release_pose

    def get_post_release_pose(self, release_pose):
        """
        Get post release pose

        Args:
            release_pose (np.array)
        
        Returns:
            post_release_pose (np.array)
        """
        post_release_pose = np.eye(4)
        post_release_pose[:3, :3] = release_pose[:3, :3] 
        post_release_pose[:3, 3] = release_pose[:3, 3] - self.retreat_distance * release_pose[:3,2]
        return post_release_pose

    def generate_supports(
        self,
        obj_mesh_on_sup,
        obj_pose_on_sup,
        n_samples_on_sup,
        obj_mesh_for_sup,
        obj_pose_for_sup,
        n_samples_for_sup,
        n_trials=1
    ):
        """
        Generate support poses

        Args:
            obj_mesh_on_sup (trimesh.base.Trimesh): mesh of support object
            obj_pose_on_sup (np.array): pose of support object
            n_samples_on_sup (int): number of sampling points on support object
            obj_mesh_for_sup (trimesh.base.Trimesh): mesh of grasp object
            obj_pose_for_sup (np.array): pose of grasp object            
            n_samples_for_sup (int): number of sampling points on grasp object
            n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points        
        
        Returns:
            release_pose, result_obj_pose (tuple): release pose, release pose of object
        """
        cnt = 0
        self.obj_mesh_for_sup = deepcopy(obj_mesh_for_sup)
        self.obj_mesh_on_sup = deepcopy(obj_mesh_on_sup)
        self.obj_mesh_on_sup.apply_transform(obj_pose_on_sup)
        while cnt < n_trials:
            support_points = self.sample_supports(obj_mesh_on_sup, obj_pose_on_sup, n_samples_on_sup,
                                            obj_mesh_for_sup, obj_pose_for_sup, n_samples_for_sup)
            
            for result_obj_pose, obj_pose_transformed_for_sup, point_on_sup, point_transformed in self._transform_points_on_support(support_points, obj_pose_for_sup):
                T = np.dot(obj_pose_for_sup, np.linalg.inv(obj_pose_transformed_for_sup))
                self.obj_pose_transformed_for_sup = obj_pose_transformed_for_sup
                gripper_pose_transformed = np.dot(T, self.tcp_pose)
                result_gripper_pose = np.eye(4)
                result_gripper_pose[:3, :3] = gripper_pose_transformed[:3, :3]
                result_gripper_pose[:3, 3] = gripper_pose_transformed[:3, 3] + (point_on_sup - point_transformed) + np.array([0, 0, self.release_distance])

                gripper_transformed = self.get_transformed_gripper_fk(result_gripper_pose)
                if self._collision_free(gripper_transformed, only_gripper=True):
                    release_pose = gripper_transformed[self.robot.eef_name]
                    yield release_pose, result_obj_pose
            cnt += 1

    def sample_supports(
        self,
        obj_mesh_on_sup,
        obj_pose_on_sup,
        n_samples_on_sup,
        obj_mesh_for_sup,
        obj_pose_for_sup,
        n_samples_for_sup,
    ):
        """
        sampling support poses

        Args:
            obj_mesh_on_sup (trimesh.base.Trimesh): mesh of support object
            obj_pose_on_sup (np.array): pose of support object
            n_samples_on_sup (int): number of sampling points on support object
            obj_mesh_for_sup (trimesh.base.Trimesh): mesh of grasp object
            obj_pose_for_sup (np.array): pose of grasp object            
            n_samples_for_sup (int): number of sampling points on grasp object

        Returns:
            point_on_support, normal_on_support, point_for_support, normal_for_support (tuple)
        """
        sample_points_on_support = self.generate_points_on_support(obj_mesh_on_sup, obj_pose_on_sup, n_samples_on_sup)
        sample_points_for_support = list(self.generate_points_for_support(obj_mesh_for_sup, obj_pose_for_sup, n_samples_for_sup))

        for point_on_support, normal_on_support in sample_points_on_support:
            for point_for_support, normal_for_support in sample_points_for_support:
                yield point_on_support, normal_on_support, point_for_support, normal_for_support

    def _transform_points_on_support(self, support_points, obj_pose_for_sup):
        """
        Transform from grasp object points to support object points
        
        Args:
            point_on_support, normal_on_support, point_for_support, normal_for_support (tuple)
            obj_pose_for_sup (np.array): pose of grasp object

        Returns:
            result_obj_pose, obj_pose_transformed_for_sup, point_on_sup, point_transformed (tuple)
        """
        for point_on_sup, normal_on_sup, point_for_sup, normal_for_sup in support_points:
            normal_on_sup = -normal_on_sup
            rot_mat = get_rotation_from_vectors(normal_for_sup, normal_on_sup)
            
            obj_pose_transformed_for_sup = np.eye(4)
            obj_pose_transformed_for_sup[:3, :3] = np.dot(rot_mat, obj_pose_for_sup[:3, :3])
            obj_pose_transformed_for_sup[:3, 3] = obj_pose_for_sup[:3, 3]

            point_transformed = np.dot(point_for_sup - obj_pose_for_sup[:3, 3], rot_mat) + obj_pose_for_sup[:3, 3]

            result_obj_pose = np.eye(4)
            result_obj_pose[:3, :3] = obj_pose_transformed_for_sup[:3, :3]
            result_obj_pose[:3, 3] = obj_pose_for_sup[:3, 3] + (point_on_sup - point_transformed) + np.array([0, 0, self.release_distance])

            yield result_obj_pose, obj_pose_transformed_for_sup, point_on_sup, point_transformed

    def filter_supports(self, support_poses):
        """
        Filter support poses

        Args:
            support_poses (tuple): release_pose, result_obj_pose
        
        Returns:
            release_pose (np.array): release pose
        """
        is_success_filtered = False
        for release_pose, result_obj_pose in support_poses:
            if not self._check_support(result_obj_pose):
                continue

            qpos = self._compute_inverse_kinematics(release_pose)
            transforms = self.robot.forward_kin(np.array(qpos))
            goal_pose = transforms[self.robot.eef_name].h_mat

            if self.has_obj:
                self.robot_c_manager.set_transform(self.obj_info["name"], result_obj_pose)

            if self._check_ik_solution(release_pose, goal_pose) and self._collision_free(transforms):
                pre_release_pose = self.get_pre_release_pose(release_pose)
                pre_release_transforms, pre_release_goal_pose = self._get_goal_pose(pre_release_pose)

                if self.has_obj:
                    obj_pre_release_pose = np.dot(pre_release_pose, self.T_between_gripper_and_obj)
                    self.robot_c_manager.set_transform(self.obj_info["name"], obj_pre_release_pose)
                    self.obj_pre_release_pose = obj_pre_release_pose

                if self._check_ik_solution(pre_release_pose, pre_release_goal_pose) and self._collision_free(pre_release_transforms):
                    self.pre_release_pose = pre_release_pose
                    
                    post_release_pose = self.get_post_release_pose(release_pose)
                    post_release_transforms, post_release_goal_pose = self._get_goal_pose(post_release_pose)

                    if self.has_obj:
                        self.object_c_manager.set_transform(self.obj_info["name"], result_obj_pose)

                    if not self._check_between_object_distances(eps=0.03):
                        continue

                    if self._check_ik_solution(post_release_pose, post_release_goal_pose) and self._collision_free(post_release_transforms):
                        self.post_release_pose = post_release_pose
                        self.obj_release_pose = result_obj_pose
                        self.obj_post_release_pose = result_obj_pose
                        is_success_filtered = True
                        break

        if not is_success_filtered:
            logger.error(f"Failed to filter Release poses")
            return None
        
        if self.has_obj:
            self.robot_c_manager.remove_object(self.obj_info["name"])

        logger.info(f"Success to get Release pose.\n")
        return release_pose

    def generate_points_on_support(
        self,
        obj_mesh,
        obj_pose,
        n_samples,
        alpha=0.99
    ):
        """
        Generate support points

        Args:
            obj_mesh (trimesh.base.Trimesh): mesh of support object
            obj_pose (np.array): pose of support object
            n_samples (int): number of sampling points on support object
            alpha (float)

        Returns:
            point, normal_vector (tuple)
        """
        copied_mesh = deepcopy(obj_mesh)
        copied_mesh.apply_transform(obj_pose)
        center_point = copied_mesh.center_mass

        weights = np.zeros(len(copied_mesh.faces))
        for idx, vertex in enumerate(copied_mesh.vertices[copied_mesh.faces]):
            weights[idx]=0.0
            if np.all(vertex[:,2] >= copied_mesh.bounds[1][2] * 0.99):
                weights[idx] = 1.0

        support_points, _, normal_vectors = surface_sampling(copied_mesh, n_samples, weights)
        for point, normal_vector in zip(support_points, normal_vectors):
            len_x = abs(center_point[0] - copied_mesh.bounds[0][0])
            len_y = abs(center_point[1] - copied_mesh.bounds[0][1])
            if center_point[0] - len_x * alpha <= point[0] <= center_point[0] + len_x * alpha:
                if center_point[1] - len_y * alpha <= point[1] <= center_point[1] + len_y * alpha:
                    yield point, normal_vector

    def generate_points_for_support(
        self,
        obj_mesh,
        obj_pose,
        n_samples
    ):
        """
        Generate grasp object point placed in support pose

        Args:
            obj_mesh (trimesh.base.Trimesh): mesh of grasp object
            obj_pose (np.array): pose of grasp object            
            n_samples (int): number of sampling points on grasp object

        Returns:
            point, normal_vector (tuple)
        """
        copied_mesh = deepcopy(obj_mesh)
        copied_mesh.apply_transform(obj_pose)
    
        weights = np.zeros(len(copied_mesh.faces))
        for idx, vertex in enumerate(copied_mesh.vertices[copied_mesh.faces]):
            weights[idx]=0.4
            if np.all(vertex[:,2] <= copied_mesh.bounds[0][2] * 1.02):                
                weights[idx] = 0.6
  
        support_points, _, normal_vectors = surface_sampling(copied_mesh, n_samples, weights)
        for point, normal_vector in zip(support_points, normal_vectors):
            yield point, normal_vector

    def _compute_inverse_kinematics(self, grasp_pose):
        """
        Compute inverse kinematics

        Args:
            grasp_pose (np.array): grasp pose

        Returns:
            qpos (np.array): joint positions 
        """
        eef_pose = get_pose_from_homogeneous(grasp_pose)
        qpos = self.robot.inverse_kin(np.random.randn(7), eef_pose, max_iter=500)
        return qpos

    def _get_goal_pose(self, pose):
        """
        Get goal pose

        Args:
            pose (np.array): current eef pose
        
        Returns:
            fk, goal_pose (tuple): forward kinematics, goal eef pose 
        """
        qpos = self._compute_inverse_kinematics(pose)
        fk = self.robot.forward_kin(np.array(qpos))
        goal_pose = fk[self.robot.eef_name].h_mat
        return fk, goal_pose

    def _check_support(self, obj_pose):
        """
        Check support pose

        Args:
            obj_pose (np.array): pose of support object
        
        Returns:
            bool: If satisfy support pose, then true
                  Otherwise then false
        """
        obj_mesh = deepcopy(self.obj_mesh_for_sup)
        obj_mesh.apply_transform(obj_pose)
        self.obj_center_point = obj_mesh.center_mass
        locations, _, _ = self.obj_mesh_on_sup.ray.intersects_location(
                    ray_origins=[self.obj_center_point],
                    ray_directions=[[0, 0, -1]])

        if len(locations) != 0:
            support_index = np.where(locations == np.max(locations, axis=0)[2])
            self.obj_support_point = locations[support_index[0]]
            return True
        logger.warning("Not found support point")
        return False

    def _check_between_object_distances(self, eps=0.01):
        distance_info = self.object_c_manager.get_distances_internal()
        distances = []
        for (o1, o2), distance in distance_info.items():
            if o1 == self.obj_info["name"] and o2 in list(self.object_c_manager.objects.grasp_objects.keys()):
                distances.append(distance)
        return np.all(eps <= np.array(distances))