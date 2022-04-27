import numpy as np
from enum import Enum, auto
from copy import deepcopy

import pykin.utils.action_utils as a_utils
from pykin.action.activity import ActivityBase
from pykin.scene.scene import Scene, SceneManager

class ReleaseStatus(Enum):
    """
    Release Status Enum class
    """
    PRE_RELEASE = auto()
    RELEASE = auto()
    POST_RELEASE = auto()

class PlaceAction(ActivityBase):
    def __init__(
        self,
        scene_mngr:SceneManager,
        n_samples_held_obj=10,
        n_samples_support_obj=10,
        release_distance=0.01
    ):
        super().__init__(scene_mngr)
        self.n_samples_held_obj = n_samples_held_obj
        self.n_samples_sup_obj = n_samples_support_obj
        self.release_distance = release_distance
        self.filter_logical_states = [scene_mngr.scene.state.held]
                                    #   scene_mngr.scene.state.static]

    def get_possible_actions(self, scene:Scene=None, level=0):
        if not 0 <= level <= 2:
            raise ValueError("Check level number!!")

        if scene is None:
            scene = self.scene_mngr.scene

        self.scene_mngr = self.scene_mngr.copy_scene(self.scene_mngr)
        self.scene_mngr.scene = deepcopy(scene)
        # self.scene_mngr.show_logical_states()

        for held_obj in self.scene_mngr.scene.objs:
            # Absolutely Need held logical state
            if self.scene_mngr.scene.logical_states[held_obj].get('held'):
                tcp_pose = self.scene_mngr.scene.robot.gripper.get_gripper_tcp_pose()
                
                for sup_obj in self.scene_mngr.scene.objs:
                    if sup_obj == held_obj:
                        continue
                    if not any(logical_state in self.scene_mngr.scene.logical_states[sup_obj] for logical_state in self.filter_logical_states):
                        release_poses = list(self.get_release_poses(sup_obj, held_obj, tcp_pose))
                        action = self.get_action(held_obj, sup_obj, release_poses)
                        if level == 0:
                            yield action
                        elif level <= 2:
                            release_poses_for_only_gripper = list(self.get_release_poses_for_only_gripper(release_poses))
                            action_level_1 = self.get_action(held_obj, sup_obj, release_poses_for_only_gripper)
                            if level == 1:
                                yield action_level_1
                            else:
                                goal_release_poses = list(self.get_release_poses_for_robot(release_poses_for_only_gripper))
                                action_level_2 = self.get_action(held_obj, sup_obj, goal_release_poses)
                                yield action_level_2

    def get_action(self, held_obj_name, PLACE_OBJ_NAME, poses):
        action = {}
        action[self.action_info.ACTION] = "place"
        action[self.action_info.HELD_OBJ_NAME] = held_obj_name
        action[self.action_info.PLACE_OBJ_NAME] = PLACE_OBJ_NAME
        action[self.action_info.RELEASE_POSES] = poses
        return action
    
    def get_possible_transitions(self, scene:Scene=None, action:dict={}):
        if not action:
            ValueError("Not found any action!!")

        held_obj = action[self.action_info.PICK_OBJ_NAME]

        for release_pose, obj_pose in action[self.action_info.RELEASE_POSES]:
            next_scene = deepcopy(scene)
            
            # Clear logical_state of held obj

            # Add logical_state of held obj : {'on' : sup_obj}
            yield next_scene

    # Not consider collision
    def get_release_poses(self, support_obj_name, held_obj_name, gripper_tcp_pose=None):
        gripper = self.scene_mngr.scene.robot.gripper
        transformed_tcp_poses = list(self.get_transformed_tcp_poses(support_obj_name, held_obj_name, gripper_tcp_pose))
        for tcp_pose, obj_pose_transformed in transformed_tcp_poses:
            release_pose = tcp_pose
            if gripper_tcp_pose is not None:
                release_pose = gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
            yield release_pose, obj_pose_transformed

    # for level wise - 1 (Consider gripper collision)
    def get_release_poses_for_only_gripper(self, release_poses):
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        if release_poses[0][0] is None:
            raise ValueError("Not found release poses!!")

        for release_pose, obj_pose_transformed in release_poses:
            self.scene_mngr.set_gripper_pose(release_pose)
            if not self._collide(is_only_gripper=True):
                yield release_pose, obj_pose_transformed

    # for level wise - 2 (Consider IK and collision)
    def get_release_poses_for_robot(self, release_poses_for_only_grpper):
        if self.scene_mngr.scene.robot is None:
            raise ValueError("Robot needs to be added first")

        release_poses = release_poses_for_only_grpper
        if not release_poses:
            return

        for release_pose, obj_pose_transformed in release_poses:
            thetas = self.scene_mngr.compute_ik(pose=release_pose, max_iter=100)
            self.scene_mngr.set_robot_eef_pose(thetas)
            release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()

            if self._solve_ik(release_pose, release_pose_from_ik) and not self._collide(is_only_gripper=False):
                yield release_pose, obj_pose_transformed

    def get_surface_points_for_support_obj(self, obj_name):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)

        weights = self._get_weights_for_support_obj(copied_mesh)
        sample_points, normals = self.get_surface_points_from_mesh(copied_mesh, self.n_samples_sup_obj, weights)
        return sample_points, normals

    @staticmethod
    def _get_weights_for_support_obj(obj_mesh):
        # heuristic
        weights = np.zeros(len(obj_mesh.faces))
        for idx, vertex in enumerate(obj_mesh.vertices[obj_mesh.faces]):
            weights[idx]=0.0
            if np.all(vertex[:,2] >= obj_mesh.bounds[1][2] * 0.99):
                weights[idx] = 1.0
        return weights

    def get_surface_points_for_held_obj(self, obj_name):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)
        
        weights = self._get_weights_for_held_obj(copied_mesh)
        sample_points, normals = self.get_surface_points_from_mesh(copied_mesh, self.n_samples_held_obj, weights)
        return sample_points, normals

    @staticmethod
    def _get_weights_for_held_obj(obj_mesh):
        # heuristic
        weights = np.zeros(len(obj_mesh.faces))
        for idx, vertex in enumerate(obj_mesh.vertices[obj_mesh.faces]):
            weights[idx]=0.3
            if np.all(vertex[:,2] <= obj_mesh.bounds[0][2] * 1.02):                
                weights[idx] = 0.7
        return weights

    def get_transformed_tcp_poses(self, support_obj_name, held_obj_name, gripper_tcp_pose=None):
        held_obj_pose = deepcopy(self.scene_mngr.scene.objs[held_obj_name].h_mat)

        support_obj_points, support_obj_normals = self.get_surface_points_for_support_obj(support_obj_name)
        held_obj_points, held_obj_normals = self.get_surface_points_for_held_obj(held_obj_name)

        for support_obj_point, support_obj_normal in zip(support_obj_points, support_obj_normals):
            for held_obj_point, held_obj_normal in zip(held_obj_points, held_obj_normals):
                rot_mat = a_utils.get_rotation_from_vectors(held_obj_normal, -support_obj_normal)
                held_obj_point_transformed = np.dot(held_obj_point - held_obj_pose[:3, 3], rot_mat) + held_obj_pose[:3, 3]
                
                held_obj_pose_transformed, held_obj_pose_rotated = self._get_obj_pose_transformed(
                    held_obj_pose, support_obj_point, held_obj_point_transformed, rot_mat)

                if gripper_tcp_pose is not None:
                    T_obj_pose_and_obj_pose_transformed = np.dot(held_obj_pose, np.linalg.inv(held_obj_pose_rotated))
                    tcp_pose_transformed = self._get_tcp_pose_transformed(
                        T_obj_pose_and_obj_pose_transformed, gripper_tcp_pose, support_obj_point, held_obj_point_transformed)
                    yield tcp_pose_transformed, held_obj_pose_transformed
                else:
                    yield None, held_obj_pose_transformed

    def _get_obj_pose_transformed(self, held_obj_pose, sup_obj_point, held_obj_point_transformed, rot_mat):
        obj_pose_rotated = np.eye(4)
        obj_pose_rotated[:3, :3] = np.dot(rot_mat, held_obj_pose[:3, :3])
        obj_pose_rotated[:3, 3] = held_obj_pose[:3, 3]

        obj_pose_transformed = np.eye(4)
        obj_pose_transformed[:3, :3] = obj_pose_rotated[:3, :3]
        obj_pose_transformed[:3, 3] = held_obj_pose[:3, 3] + (sup_obj_point - held_obj_point_transformed) + np.array([0, 0, self.release_distance])
        return obj_pose_transformed, obj_pose_rotated

    def _get_tcp_pose_transformed(self, T, tcp_pose, sup_obj_point, held_obj_point_transformed):
        tcp_pose_transformed = np.dot(T, tcp_pose)

        result_tcp_pose_transformed = np.eye(4)
        result_tcp_pose_transformed[:3, :3] = tcp_pose_transformed[:3, :3]
        result_tcp_pose_transformed[:3, 3] = tcp_pose_transformed[:3, 3] + (sup_obj_point - held_obj_point_transformed) + np.array([0, 0, self.release_distance])
        return result_tcp_pose_transformed