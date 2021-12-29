import numpy as np
from copy import deepcopy

from pykin.tasks.activity import ActivityBase
from pykin.utils.task_utils import normalize, surface_sampling, projection

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


    def generate_eef_poses(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        n_steps=10,
        n_trials=10
    ):
        tcp_poses = self.generate_tcp_poses(obj_mesh, obj_pose, limit_angle, n_steps, n_trials)
        for tcp_pose, contact_point in tcp_poses:
            eef_pose = self.get_eef_h_mat_from_tcp(tcp_pose)
            yield eef_pose, tcp_pose, contact_point

    def generate_tcp_poses(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        n_steps,
        n_trials
    ):
        for contact_points, _ in self._generate_contact_points(obj_mesh, obj_pose, limit_angle, n_steps):
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

                yield tcp_pose, contact_points

    def _generate_contact_points(
        self,
        obj_mesh,
        obj_pose,
        limit_angle,
        n_steps
    ):
        copied_mesh = deepcopy(obj_mesh)
        copied_mesh.apply_transform(obj_pose)

        cnt = 0
        while cnt < n_steps:
            contact_points, _, normals = surface_sampling(copied_mesh, n_samples=2)
            if self._is_force_closure(contact_points, normals, limit_angle):
                cnt += 1
                yield (contact_points, normals)

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

    def _generate_grasp_directions(self, vector, n_steps=10):
        norm_vector = normalize(vector)
        e1, e2 = np.eye(3)[:2]
        v1 = e1 - projection(e1, norm_vector)
        v1 = normalize(v1)
        v2 = e2 - projection(e2, norm_vector) - projection(e2, v1)
        v2 = normalize(v2)

        for theta in np.linspace(-np.pi/2, np.pi/2, n_steps):
            normal_dir = np.cos(theta) * v1 + np.sin(theta) * v2
            yield normal_dir

    def filter_grasp_poses(self):
        pass

    def get_grasp_waypoints(self):
        # pregrasp
        # grasp
        # retreat
        pass

    def get_pre_grasp_pose(self):
        pass

    def get_grasp_pose(self):
        pass

    def get_post_grasp_pose(self):
        pass

    def collision_free(self) -> bool:
        return True

    def _check_ik_solution(self) -> bool:
        return False


    