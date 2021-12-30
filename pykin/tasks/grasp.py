import numpy as np
from collections import OrderedDict
from copy import deepcopy

from pykin.tasks.activity import ActivityBase
from pykin.utils.task_utils import normalize, surface_sampling, projection
from pykin.utils.transform_utils import get_pose_from_homogeneous
from pykin.utils.log_utils import create_logger

logger = create_logger('Grasp', "debug")

class Grasp(ActivityBase):
    def __init__(
        self,
        robot,
        robot_col_manager,
        obstacles_col_manager,
        mesh_path,
        **gripper_configures
    ):
        super().__init__(
            robot,
            robot_col_manager,
            obstacles_col_manager,
            mesh_path,
            **gripper_configures)

    def get_grasp_waypoints(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        num_grasp=1,
        n_trials=1,
        desired_distance=0.10
    ):
        waypoints = OrderedDict()

        grasp_pose, _, _ = self.get_grasp_pose(obj_mesh, obj_pose, limit_angle, num_grasp, n_trials, desired_distance)
        
        waypoints["pre_grasp"] = self.pre_goal_pose
        waypoints["grasp"] = grasp_pose
        waypoints["post_grasp"] =self.post_grasp_pose

        return waypoints
        
    def get_pre_grasp_pose(self, grasp_pose, desired_distance):
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = grasp_pose[:3, :3]
        pre_grasp_pose[:3, 3] = grasp_pose[:3, 3] - desired_distance * grasp_pose[:3,2]    
        return pre_grasp_pose

    def get_grasp_pose(        
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        num_grasp=1,
        n_trials=1,
        desired_distance=0.1
    ):
        grasp_poses = list(self.generate_grasps(obj_mesh, obj_pose, limit_angle, num_grasp, n_trials))
        grasp_pose, tcp_pose, contact_point = self.filter_grasps(grasp_poses, n_trials, desired_distance)
        return grasp_pose, tcp_pose, contact_point

    def generate_grasps(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        num_grasp=1,
        n_trials=1
    ):
        cnt = 0
        gripper = self.get_gripper()
        while cnt < num_grasp * n_trials:
            tcp_poses = self._generate_tcp_poses(obj_mesh, obj_pose, limit_angle, n_trials)
            for tcp_pose, contact_point in tcp_poses:
                eef_pose = self.get_eef_h_mat_from_tcp(tcp_pose)
                gripper_transformed = self.get_gripper_transformed(gripper, tcp_pose)

                if self.collision_free(gripper_transformed, only_gripper=True):
                    # cnt += 1
                    yield (eef_pose, tcp_pose, contact_point)
            cnt += 1

    def filter_grasps(self, grasp_poses, n_trials=0, desired_distance=0.1):
        is_success_filtered = False
        for grasp_pose, tcp_pose, contact_point in grasp_poses:
            qpos = self._compute_inverse_kinematics(grasp_pose, n_trials)
            if qpos is None:
                continue

            transforms = self.robot.forward_kin(np.array(qpos))
            goal_pose = transforms[self.robot.eef_name].h_mat
 
            if self._check_ik_solution(grasp_pose, goal_pose) and self.collision_free(transforms):
                pre_grasp_pose = self.get_pre_grasp_pose(grasp_pose, desired_distance)
                pre_qpos = self._compute_inverse_kinematics(pre_grasp_pose, 5)
                pre_transforms = self.robot.forward_kin(np.array(pre_qpos))
                pre_goal_pose = pre_transforms[self.robot.eef_name].h_mat

                if self._check_ik_solution(pre_grasp_pose, pre_goal_pose) and self.collision_free(pre_transforms):
                    self.pre_goal_pose = pre_grasp_pose
                    self.post_grasp_pose = pre_grasp_pose
                    is_success_filtered = True
                    break

        if not is_success_filtered:
            logger.error(f"Failed to filter grasp poses")
            return None, None, None

        return grasp_pose, tcp_pose, contact_point

    def _compute_inverse_kinematics(self, grasp_pose, n_trials):
        eef_pose = get_pose_from_homogeneous(grasp_pose)
        qpos = self.robot.inverse_kin(np.random.randn(7), eef_pose, maxIter=500)
        return qpos

    def _generate_tcp_poses(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        n_trials
    ):
        contact_points, _ = self._generate_contact_points(obj_mesh, obj_pose, limit_angle)
        p1, p2 = contact_points
        center_point = (p1 + p2) /2
        line = p2 - p1

        for i, grasp_dir in enumerate(self._generate_grasp_directions(line, n_trials)):
            y = normalize(line)
            z = grasp_dir
            x = np.cross(y, z)

            tcp_pose = np.eye(4)
            tcp_pose[:3,0] = x
            tcp_pose[:3,1] = y
            tcp_pose[:3,2] = z
            tcp_pose[:3,3] = center_point

            yield (tcp_pose, contact_points)

    def _generate_contact_points(
        self,
        obj_mesh,
        obj_pose,
        limit_angle
    ):
        copied_mesh = deepcopy(obj_mesh)
        copied_mesh.apply_transform(obj_pose)

        while True:
            contact_points, _, normals = surface_sampling(copied_mesh, n_samples=2)
            if self._is_force_closure(contact_points, normals, limit_angle):
                break
        return (contact_points, normals)

    def _is_force_closure(self, vertices, normals, limit_angle):
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

    def _generate_grasp_directions(self, vector, n_trials):
        norm_vector = normalize(vector)
        e1, e2 = np.eye(3)[:2]
        v1 = e1 - projection(e1, norm_vector)
        v1 = normalize(v1)
        v2 = e2 - projection(e2, norm_vector) - projection(e2, v1)
        v2 = normalize(v2)

        for theta in np.linspace(-np.pi/2, np.pi/2, n_trials):
            normal_dir = np.cos(theta) * v1 + np.sin(theta) * v2
            yield normal_dir

    def _check_ik_solution(self, eef_pose, goal_pose, err_limit=1e-2) -> bool:
        error_pose = self.robot.get_pose_error(eef_pose, goal_pose)
        if error_pose < err_limit:
            return True
        return False