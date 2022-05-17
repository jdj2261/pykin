import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from trimesh import Trimesh, proximity

import pykin.utils.action_utils as a_utils
from pykin.action.activity import ActivityBase
from pykin.scene.scene import Scene
from pykin.utils.log_utils import create_logger

logger = create_logger('PlaceAction', "debug")

@dataclass
class ReleaseName:
    """
    Release Status Enum class
    """
    PRE_RELEASE = "pre_release"
    RELEASE = "release"
    POST_RELEASE = "post_release"

class PlaceAction(ActivityBase):
    def __init__(
        self,
        scene_mngr,
        n_samples_held_obj=10,
        n_samples_support_obj=10,
        release_distance=0.01,
    ):
        super().__init__(scene_mngr)
        self.release_name = ReleaseName
        self.n_samples_held_obj = n_samples_held_obj
        self.n_samples_sup_obj = n_samples_support_obj
        self.release_distance = release_distance
        self.filter_logical_states = [scene_mngr.scene.state.held]                                    

    def get_possible_actions_level_1(self, scene:Scene=None) -> dict:
        if scene is None:
            scene = self.scene_mngr.scene

        self.scene_mngr.scene = deepcopy(scene)

        held_obj = self.scene_mngr.scene.robot.gripper.attached_obj_name
        eef_pose = self.scene_mngr.scene.robot.gripper.grasp_pose
        self.scene_mngr.scene.objs[held_obj].h_mat = self.scene_mngr.scene.robot.gripper.pick_obj_pose
        
        for sup_obj in deepcopy(self.scene_mngr.scene.objs):
            if sup_obj == held_obj:
                continue

            if sup_obj == self.scene_mngr.scene.place_obj_name:
                if not "table" in sup_obj:
                    continue

            if not any(logical_state in self.scene_mngr.scene.logical_states[sup_obj] for logical_state in self.filter_logical_states):
                release_poses = list(self.get_all_release_poses(sup_obj, held_obj, eef_pose))
                release_poses_for_only_gripper = list(self.get_release_poses_for_only_gripper(release_poses))
                action_level_1 = self.get_action(held_obj, sup_obj, release_poses_for_only_gripper)
                yield action_level_1

    # Not Expand, only check possible action using ik
    def get_possible_ik_solve_level_2(self, scene:Scene=None, release_poses:dict={}) -> bool:
        if scene is None:
            scene = self.scene_mngr.scene
        self.scene_mngr.scene = deepcopy(scene)
        
        ik_solve, release_poses_filtered = self.compute_ik_solve_for_robot(release_poses)
        return ik_solve, release_poses_filtered

    def get_possible_joint_path_level_3(self, scene:Scene=None, release_poses:dict={}, init_thetas=None):
        if scene is None:
            scene = self.scene_mngr.scene
        self.scene_mngr.scene = deepcopy(scene)

        result_all_joint_path = []
        result_joint_path = {}
        default_joint_path = []

        default_thetas = init_thetas
        if init_thetas is None:
            default_thetas = self.scene_mngr.scene.robot.init_qpos
        
        pre_release_pose = release_poses[self.release_name.PRE_RELEASE]
        release_pose = release_poses[self.release_name.RELEASE]
        post_release_pose = release_poses[self.release_name.POST_RELEASE]
        success_joint_path = False
        # default pose -> pre_release_pose (rrt)
        self.scene_mngr.set_robot_eef_pose(default_thetas)
        
        self.scene_mngr.set_object_pose(scene.pick_obj_name, scene.pick_obj_default_pose)
        self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name)
        pre_release_joint_path = self.get_rrt_star_path(default_thetas, pre_release_pose)
        if pre_release_joint_path:
            # pre_release_pose -> release_pose (cartesian)
            release_joint_path = self.get_cartesian_path(pre_release_joint_path[-1], release_pose)
            if release_joint_path:
                success_joint_path = True
                self.scene_mngr.detach_object_from_gripper()
                self.scene_mngr.add_object(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                    scene.robot.gripper.place_obj_pose,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)

                # release_pose -> post_release_pose (cartesian)
                post_release_joint_path = self.get_cartesian_path(release_joint_path[-1], post_release_pose)
                if post_release_joint_path:
                    # post_release_pose -> default pose (rrt)
                    default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)["right_gripper"].h_mat
                    default_joint_path = self.get_rrt_star_path(post_release_joint_path[-1], default_pose)

        if not success_joint_path:
            self.scene_mngr.detach_object_from_gripper()
            self.scene_mngr.add_object(
                self.scene_mngr.scene.robot.gripper.attached_obj_name,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                scene.robot.gripper.place_obj_pose,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)

        if default_joint_path:
            result_joint_path.update({self.release_name.PRE_RELEASE: pre_release_joint_path})
            result_joint_path.update({self.release_name.RELEASE: release_joint_path})
            result_joint_path.update({self.release_name.POST_RELEASE: post_release_joint_path})
            result_joint_path.update({"default_pose": default_joint_path})
            result_all_joint_path.append(result_joint_path)
        
            return result_all_joint_path

    def get_action(self, held_obj_name, place_obj_name, poses):
        action = {}
        action[self.action_info.ACTION] = "place"
        action[self.action_info.HELD_OBJ_NAME] = held_obj_name
        action[self.action_info.PLACE_OBJ_NAME] = place_obj_name
        action[self.action_info.RELEASE_POSES] = poses
        return action
    
    def get_possible_transitions(self, scene:Scene=None, action:dict={}):
        if not action:
            ValueError("Not found any action!!")

        held_obj_name = action[self.action_info.HELD_OBJ_NAME]
        place_obj_name = action[self.action_info.PLACE_OBJ_NAME]

        for release_poses, obj_pose_transformed in action[self.action_info.RELEASE_POSES]:
            next_scene = deepcopy(scene)
            
            ## Change transition
            next_scene.release_poses = release_poses
            next_scene.robot.gripper.place_obj_pose = obj_pose_transformed
            next_scene.robot.gripper.release_pose = release_poses[self.release_name.RELEASE]
        
            # Move a gripper to default pose
            default_thetas = self.scene_mngr.scene.robot.init_qpos
            default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)["right_gripper"].h_mat
            next_scene.robot.gripper.set_gripper_pose(default_pose)

            # Move pick object on support obj

            next_scene.objs[held_obj_name].h_mat = obj_pose_transformed
            next_scene.pick_obj_name = held_obj_name
            ## Change Logical State
            # Clear logical_state of held obj
            next_scene.logical_states.get(held_obj_name).clear()

            # Chage logical_state holding : None
            next_scene.logical_states[next_scene.robot.gripper.name][next_scene.state.holding] = None

            # Add logical_state of held obj : {'on' : place_obj}
            next_scene.logical_states[held_obj_name][next_scene.state.on] = next_scene.objs[place_obj_name]
            next_scene.update_logical_states()


            copied_scene = deepcopy(next_scene)
            held_obj = copied_scene.objs[held_obj_name]
            held_obj_mesh:Trimesh = deepcopy(held_obj.gparam)
            held_obj_mesh.apply_transform(obj_pose_transformed)
            com = held_obj_mesh.center_mass

            if not self._check_stability(next_scene, held_obj_name, com):
                pass

            yield next_scene

    # Not consider collision
    def get_all_release_poses(self, support_obj_name, held_obj_name, gripper_eef_pose=None):
        # gripper = self.scene_mngr.scene.robot.gripper
        transformed_eef_poses = list(self.get_transformed_eef_poses(support_obj_name, held_obj_name, gripper_eef_pose))
        
        for eef_pose, obj_pose_transformed in transformed_eef_poses:
            if not self._check_support(support_obj_name, held_obj_name, obj_pose_transformed):
                continue
            
            release_pose = {}
            release_pose[self.release_name.RELEASE] = eef_pose
            release_pose[self.release_name.PRE_RELEASE] = self.get_pre_release_pose(eef_pose)
            release_pose[self.release_name.POST_RELEASE] = self.get_post_release_pose(eef_pose)
            yield release_pose, obj_pose_transformed

    def get_pre_release_pose(self, release_pose):
        pre_release_pose = np.eye(4)
        pre_release_pose[:3, :3] = release_pose[:3, :3]
        pre_release_pose[:3, 3] = release_pose[:3, 3] + np.array([0, 0, self.retreat_distance])
        return pre_release_pose

    def get_post_release_pose(self, release_pose):
        post_release_pose = np.eye(4)
        post_release_pose[:3, :3] = release_pose[:3, :3] 
        post_release_pose[:3, 3] = release_pose[:3, 3] - self.retreat_distance * release_pose[:3,2]
        return post_release_pose

    # for level wise - 1 (Consider gripper collision)
    def get_release_poses_for_only_gripper(self, release_poses, is_attached=True):
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        for all_release_pose, obj_pose_transformed in release_poses:
            if is_attached:
                self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name)
            for name, pose in all_release_pose.items():
                is_collision = False
                if name == self.release_name.RELEASE:
                    self.scene_mngr.set_gripper_pose(pose)
                    for name in self.scene_mngr.scene.objs:
                        self.scene_mngr.obj_collision_mngr.set_transform(name, self.scene_mngr.scene.objs[name].h_mat)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.release_name.PRE_RELEASE:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.release_name.POST_RELEASE:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
            
            if is_attached:
                self.scene_mngr.detach_object_from_gripper()
                self.scene_mngr.add_object(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                    self.scene_mngr.scene.robot.gripper.pick_obj_pose,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)

            if not is_collision:
                yield all_release_pose, obj_pose_transformed

    def compute_ik_solve_for_robot(self, release_pose:dict, is_attached=True):
        ik_sovle = {}
        release_pose_for_ik = {}

        if is_attached:
            self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name)
        
        for name, pose in release_pose.items():
            if name == self.release_name.RELEASE:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, release_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_sovle[name] = thetas
                    release_pose_for_ik[name] = pose
            if name == self.release_name.PRE_RELEASE:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                pre_release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, pre_release_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_sovle[name] = thetas
                    release_pose_for_ik[name] = pose
            if name == self.release_name.POST_RELEASE:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                post_release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, post_release_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_sovle[name] = thetas
                    release_pose_for_ik[name] = pose

        if is_attached:
            self.scene_mngr.detach_object_from_gripper()
            self.scene_mngr.add_object(
                self.scene_mngr.scene.robot.gripper.attached_obj_name,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                self.scene_mngr.scene.robot.gripper.pick_obj_pose,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)

        if len(ik_sovle) == 3:
            return ik_sovle, release_pose_for_ik
        return None, None

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
        copied_mesh = deepcopy(self.scene_mngr.init_objects[obj_name].gparam)
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

    def get_transformed_eef_poses(self, support_obj_name, held_obj_name, gripper_eef_pose=None):
        held_obj_pose = deepcopy(self.scene_mngr.scene.objs[held_obj_name].h_mat)

        support_obj_points, support_obj_normals = self.get_surface_points_for_support_obj(support_obj_name)
        held_obj_points, held_obj_normals = self.get_surface_points_for_held_obj(held_obj_name)

        for support_obj_point, support_obj_normal in zip(support_obj_points, support_obj_normals):
            for held_obj_point, held_obj_normal in zip(held_obj_points, held_obj_normals):
                rot_mat = a_utils.get_rotation_from_vectors(held_obj_normal, -support_obj_normal)
                held_obj_point_transformed = np.dot(held_obj_point - held_obj_pose[:3, 3], rot_mat) + held_obj_pose[:3, 3]
                
                held_obj_pose_transformed, held_obj_pose_rotated = self._get_obj_pose_transformed(
                    held_obj_pose, support_obj_point, held_obj_point_transformed, rot_mat)

                if gripper_eef_pose is not None:
                    T_obj_pose_and_obj_pose_transformed = np.dot(held_obj_pose, np.linalg.inv(held_obj_pose_rotated))
                    eef_pose_transformed = self._get_eef_pose_transformed(
                        T_obj_pose_and_obj_pose_transformed, gripper_eef_pose, support_obj_point, held_obj_point_transformed)
                    yield eef_pose_transformed, held_obj_pose_transformed
                else:
                    yield None, held_obj_pose_transformed

    def _get_obj_pose_transformed(self, held_obj_pose, sup_obj_point, held_obj_point_transformed, rot_mat):
        obj_pose_rotated = np.eye(4)
        obj_pose_rotated[:3, :3] = np.dot(rot_mat, held_obj_pose[:3, :3])
        obj_pose_rotated[:3, 3] = held_obj_pose[:3, 3]

        obj_pose_transformed = np.eye(4)
        obj_pose_transformed[:3, :3] = obj_pose_rotated[:3, :3]
        obj_pose_transformed[:3, 3] = held_obj_pose[:3, 3] + (sup_obj_point - held_obj_point_transformed)
        return obj_pose_transformed, obj_pose_rotated

    def _get_eef_pose_transformed(self, T, eef_pose, sup_obj_point, held_obj_point_transformed):
        eef_pose_transformed = np.dot(T, eef_pose)

        result_eef_pose_transformed = np.eye(4)
        result_eef_pose_transformed[:3, :3] = eef_pose_transformed[:3, :3]
        result_eef_pose_transformed[:3, 3] = eef_pose_transformed[:3, 3] + (sup_obj_point - held_obj_point_transformed) + np.array([0, 0, self.release_distance])
        return result_eef_pose_transformed

    @staticmethod
    def _check_stability(copied_scene:Scene, held_obj_name, com):
        if copied_scene.state.on in list(copied_scene.logical_states[held_obj_name].keys()):
            support_obj = copied_scene.logical_states[held_obj_name][copied_scene.state.on]
            support_obj_mesh:Trimesh = deepcopy(support_obj.gparam)
            support_obj_mesh.apply_transform(support_obj.h_mat)

            closest_points, dists, surface_idx = proximity.closest_point(support_obj_mesh, [com])
            closest_point = closest_points[0]
            norm_vector = a_utils.normalize(closest_point - com)

            safe_com = norm_vector[2] == -1.0
            if safe_com:
                return PlaceAction._check_stability(copied_scene, support_obj.name, com)
            else:
                logger.warning("Not satisfied stability")
                return False
        else:
            return True

    def _check_support(self, support_obj_name, held_obj_name, obj_pose):
        """
        Check support pose
        Args:
            obj_pose (np.array): pose of support object
        
        Returns:
            bool: If satisfy support pose, then true
                  Otherwise then false
        """
        held_obj = self.scene_mngr.scene.objs[held_obj_name]
        held_obj_mesh:Trimesh = deepcopy(held_obj.gparam)
        held_obj_mesh.apply_transform(obj_pose)
        
        place_obj = self.scene_mngr.scene.objs[support_obj_name]
        place_obj_mesh = deepcopy(place_obj.gparam)
        place_obj_mesh.apply_transform(place_obj.h_mat)

        locations, _, _ = place_obj_mesh.ray.intersects_location(
                    ray_origins=[held_obj_mesh.center_mass],
                    ray_directions=[[0, 0, -1]])
        if len(locations) != 0:
            return True
        # logger.warning("Not found support point")
        return False

    def _compute_com(self):
        pass