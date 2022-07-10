import numpy as np
import random
from collections import OrderedDict
from copy import deepcopy
from torch import normal
from trimesh import Trimesh, proximity

import pykin.utils.mesh_utils as m_utils
from pykin.action.activity import ActivityBase
from pykin.scene.scene import Scene
from pykin.utils.log_utils import create_logger

logger = create_logger('PlaceAction', "debug")

class PlaceAction(ActivityBase):
    def __init__(
        self,
        scene_mngr,
        n_samples_held_obj=10,
        n_samples_support_obj=10,
        release_distance=0.01,
    ):
        super().__init__(scene_mngr)
        self.n_samples_held_obj = n_samples_held_obj
        self.n_samples_sup_obj = n_samples_support_obj
        self.release_distance = release_distance
        self.filter_logical_states = [scene_mngr.scene.logical_state.held]                                    

    def get_possible_actions_level_1(self, scene:Scene=None) -> dict:
        self.deepcopy_scene(scene)
        held_obj = self.scene_mngr.scene.robot.gripper.attached_obj_name
        eef_pose = self.scene_mngr.scene.robot.gripper.grasp_pose
        self.scene_mngr.scene.objs[held_obj].h_mat = self.scene_mngr.scene.robot.gripper.pick_obj_pose
        
        for sup_obj in deepcopy(self.scene_mngr.scene.objs):
            # print(f"place : {sup_obj}")
            if sup_obj == held_obj:
                continue

            #? for benchmark 1
            if "ceiling" in sup_obj:
                continue

            #? for benchmark 2
            if "bin" in sup_obj or "bottle" in sup_obj:
                continue
        
            if self.scene_mngr.scene.bench_num == 2 and sup_obj not in ["shelf_9"]:
                continue
            
            if sup_obj == self.scene_mngr.scene.place_obj_name:
                if sup_obj not in ["table", "shelf_9", "shelf_15"]:
                    continue

            if not any(logical_state in self.scene_mngr.scene.logical_states[sup_obj] for logical_state in self.filter_logical_states):
                action_level_1 = self.get_action_level_1_for_single_object(sup_obj, held_obj, eef_pose)
                if not action_level_1[self.info.RELEASE_POSES]:
                    continue

                yield action_level_1

    def get_action_level_1_for_single_object(self, sup_obj_name, held_obj_name, eef_pose=None, scene:Scene=None):
        if scene is not None:
            scene.objs[held_obj_name].h_mat = scene.robot.gripper.pick_obj_pose
            self.deepcopy_scene(scene)
            
        release_poses = list(self.get_all_release_poses(sup_obj_name, held_obj_name, eef_pose))
        release_poses_not_collision = list(self.get_release_poses_not_collision(release_poses))
        action_level_1 = self.get_action(held_obj_name, sup_obj_name, release_poses_not_collision)
        return action_level_1

    # Not Expand, only check possible action using ik
    def get_possible_ik_solve_level_2(self, scene:Scene=None, release_poses:dict={}):
        self.deepcopy_scene(scene)
        
        ik_solve, release_poses_filtered = self.compute_ik_solve_for_robot(release_poses)
        return ik_solve, release_poses_filtered

    def get_possible_joint_path_level_3(self, scene:Scene=None, release_poses:dict={}, init_thetas=None):
        self.deepcopy_scene(scene)

        result_all_joint_path = []
        result_joint_path = OrderedDict()
        default_joint_path = []

        default_thetas = init_thetas
        if init_thetas is None:
            default_thetas = self.scene_mngr.scene.robot.init_qpos
        
        pre_release_pose = release_poses[self.move_data.MOVE_pre_release]
        release_pose = release_poses[self.move_data.MOVE_release]
        post_release_pose = release_poses[self.move_data.MOVE_post_release]
        success_joint_path = True

        self.scene_mngr.set_robot_eef_pose(default_thetas)        
        self.scene_mngr.set_object_pose(scene.pick_obj_name, scene.pick_obj_default_pose)
        self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name, True)
        
        pre_release_joint_path = self.get_rrt_star_path(default_thetas, pre_release_pose)
        if pre_release_joint_path:
            # pre_release_pose -> release_pose (cartesian)
            release_joint_path = self.get_cartesian_path(pre_release_joint_path[-1], release_pose)
            if release_joint_path:
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
                else:
                    success_joint_path = False  
            else:
                success_joint_path = False    
        else:
            success_joint_path = False

        if not success_joint_path:
            if self.scene_mngr.is_attached:
                self.scene_mngr.detach_object_from_gripper()
            self.scene_mngr.add_object(
                self.scene_mngr.scene.robot.gripper.attached_obj_name,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                scene.robot.gripper.place_obj_pose,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)
            return result_all_joint_path

        if default_joint_path:
            result_joint_path.update({self.move_data.MOVE_pre_release: pre_release_joint_path})
            result_joint_path.update({self.move_data.MOVE_release: release_joint_path})
            result_joint_path.update({self.move_data.MOVE_post_release: post_release_joint_path})
            result_joint_path.update({self.move_data.MOVE_default_release: default_joint_path})
            result_all_joint_path.append(result_joint_path)
        
            return result_all_joint_path

    def get_action(self, held_obj_name, place_obj_name, poses):
        action = {}
        action[self.info.TYPE] = "place"
        action[self.info.HELD_OBJ_NAME] = held_obj_name
        action[self.info.PLACE_OBJ_NAME] = place_obj_name
        action[self.info.RELEASE_POSES] = poses
        return action
    
    def get_possible_transitions(self, scene:Scene=None, action:dict={}):
        if not action:
            ValueError("Not found any action!!")

        held_obj_name = action[self.info.HELD_OBJ_NAME]
        place_obj_name = action[self.info.PLACE_OBJ_NAME]

        for release_poses, obj_pose_transformed in action[self.info.RELEASE_POSES]:
            next_scene = deepcopy(scene)
            
            ## Change transition
            next_scene.release_poses = release_poses
            next_scene.robot.gripper.place_obj_pose = obj_pose_transformed
            next_scene.robot.gripper.release_pose = release_poses[self.move_data.MOVE_release]
        
            # Move a gripper to default pose
            default_thetas = self.scene_mngr.scene.robot.init_qpos
            default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)["right_gripper"].h_mat
            next_scene.robot.gripper.set_gripper_pose(default_pose)

            # Move pick object on support obj
            next_scene.objs[held_obj_name].h_mat = obj_pose_transformed
            self.scene_mngr.obj_collision_mngr.set_transform(held_obj_name, obj_pose_transformed)
            next_scene.pick_obj_name = held_obj_name
            
            ## Change Logical State
            # Clear logical_state of held obj
            next_scene.logical_states.get(held_obj_name).clear()

            # Chage logical_state holding : None
            next_scene.logical_states[next_scene.robot.gripper.name][next_scene.logical_state.holding] = None

            # Add logical_state of held obj : {'on' : place_obj}
            next_scene.logical_states[held_obj_name][next_scene.logical_state.on] = next_scene.objs[place_obj_name]
            next_scene.update_logical_states()

            # Check stability
            copied_scene = deepcopy(next_scene)
            held_obj = copied_scene.objs[held_obj_name]
            held_obj_mesh:Trimesh = deepcopy(held_obj.gparam)
            held_obj_mesh.apply_transform(obj_pose_transformed)
            com = held_obj_mesh.center_mass

            if not self._check_stability(next_scene, held_obj_name, com) and self.scene_mngr.scene.bench_num == 1:
                continue

            yield next_scene

    # Not consider collision
    def get_all_release_poses(self, support_obj_name, held_obj_name, eef_pose=None):
        # gripper = self.scene_mngr.scene.robot.gripper
        transformed_eef_poses = list(self.get_transformed_eef_poses(support_obj_name, held_obj_name, eef_pose))
        
        for eef_pose, obj_pose_transformed in transformed_eef_poses:
            if self.scene_mngr._scene.bench_num == 1:
                if not self._check_support(support_obj_name, held_obj_name, obj_pose_transformed):
                    continue

            release_pose = {}
            release_pose[self.move_data.MOVE_release] = eef_pose
            release_pose[self.move_data.MOVE_pre_release] = self.get_pre_release_pose(eef_pose)
            release_pose[self.move_data.MOVE_post_release] = self.get_post_release_pose(eef_pose)
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
    def get_release_poses_not_collision(self, release_poses, is_attached=True):
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        for all_release_pose, obj_pose_transformed in release_poses:
            if is_attached:
                self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name, True)
            for name, pose in all_release_pose.items():
                is_collision = False
                if name == self.move_data.MOVE_release:
                    self.scene_mngr.set_gripper_pose(pose)
                    for name in self.scene_mngr.scene.objs:
                        self.scene_mngr.obj_collision_mngr.set_transform(name, self.scene_mngr.scene.objs[name].h_mat)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.move_data.MOVE_pre_release:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.move_data.MOVE_post_release:
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
        ik_solve = {}
        release_pose_for_ik = {}

        if is_attached:
            self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name, True)
        
        for name, pose in release_pose.items():
            if name == self.move_data.MOVE_release:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, release_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_solve[name] = thetas
                    release_pose_for_ik[name] = pose
            if name == self.move_data.MOVE_pre_release:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                pre_release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, pre_release_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_solve[name] = thetas
                    release_pose_for_ik[name] = pose
            if name == self.move_data.MOVE_post_release:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                post_release_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, post_release_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_solve[name] = thetas
                    release_pose_for_ik[name] = pose

        if is_attached:
            self.scene_mngr.detach_object_from_gripper()
            self.scene_mngr.add_object(
                self.scene_mngr.scene.robot.gripper.attached_obj_name,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                self.scene_mngr.scene.robot.gripper.pick_obj_pose,
                self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)

        if len(ik_solve) == 3:
            return ik_solve, release_pose_for_ik
        return None, None

    def get_surface_points_for_support_obj(self, obj_name, alpha=0.2):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)
        center_point = copied_mesh.center_mass

        len_x = abs(center_point[0] - copied_mesh.bounds[0][0])
        len_y = abs(center_point[1] - copied_mesh.bounds[0][1])

        min_x = center_point[0] - len_x * alpha
        max_x = center_point[0] + len_x * alpha
        min_y = center_point[1] - len_y * alpha
        max_y = center_point[1] + len_y * alpha
        margin = (min_x, max_x, min_y, max_y)

        weights = self._get_weights_for_support_obj(copied_mesh)
        sample_points, normals = self.get_surface_points_from_mesh(copied_mesh, self.n_samples_sup_obj, weights)
        normals = np.tile(np.array([0., 0., 1.]), (normals.shape[0],1))

        # TODO heuristic
        if obj_name not in ["table", "shelf_9"]:
            center_upper_point = np.zeros(3)
            center_upper_point[0] = center_point[0] + random.uniform(-0.002, 0.002)
            center_upper_point[1] = center_point[1] + random.uniform(-0.002, 0.002)
            center_upper_point[2] = copied_mesh.bounds[1, 2]
            sample_points = np.append(sample_points, np.array([center_upper_point]), axis=0)
            normals = np.append(normals, np.array([[0, 0, 1]]), axis=0)
            
        for point, normal_vector in zip(sample_points, normals):
            yield point, normal_vector, margin

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
        center_point = copied_mesh.center_mass

        center_upper_point = center_point
        center_upper_point[-1] = copied_mesh.bounds[0, 2]
        
        if "bottle" in obj_name:
            sample_points = np.array([center_upper_point])
            normals = np.array([[0, 0, -1]])

            for point, normal_vector in zip(sample_points, normals):
                yield point, normal_vector

        weights = self._get_weights_for_held_obj(copied_mesh)
        sample_points, normals = self.get_surface_points_from_mesh(copied_mesh, self.n_samples_held_obj, weights)

        # heuristic
        sample_points = np.append(sample_points, np.array([center_upper_point]), axis=0)
        normals = np.append(normals, np.array([[0, 0, -1]]), axis=0)

        for point, normal_vector in zip(sample_points, normals):
            yield point, normal_vector

    @staticmethod
    def _get_weights_for_held_obj(obj_mesh):
        # heuristic
        weights = np.zeros(len(obj_mesh.faces))
        for idx, vertex in enumerate(obj_mesh.vertices[obj_mesh.faces]):
            weights[idx]=0.4
            if np.all(vertex[:,2] <= obj_mesh.bounds[0][2] * 1.02):                
                weights[idx] = 0.6
        return weights

    def get_transformed_eef_poses(self, support_obj_name, held_obj_name, eef_pose=None):
        bench_num = self.scene_mngr.scene.bench_num
        if bench_num == 1:
            alpha = 0.2
        elif bench_num == 2:
            alpha = 0.8
        
        held_obj_pose = deepcopy(self.scene_mngr.scene.objs[held_obj_name].h_mat)
        surface_points_for_sup_obj = list(self.get_surface_points_for_support_obj(support_obj_name, alpha=alpha))
        surface_points_for_held_obj = list(self.get_surface_points_for_held_obj(held_obj_name))
    
        for support_obj_point, support_obj_normal, (min_x, max_x, min_y, max_y) in surface_points_for_sup_obj:
            for held_obj_point, held_obj_normal in surface_points_for_held_obj:
                rot_mat = m_utils.get_rotation_from_vectors(held_obj_normal, -support_obj_normal)
                held_obj_point_transformed = np.dot(held_obj_point - held_obj_pose[:3, 3], rot_mat) + held_obj_pose[:3, 3]
                
                held_obj_pose_transformed, held_obj_pose_rotated = self._get_obj_pose_transformed(
                    held_obj_pose, support_obj_point, held_obj_point_transformed, rot_mat)
                
                # heuristic
                copied_mesh = deepcopy(self.scene_mngr.init_objects[held_obj_name].gparam)
                copied_mesh.apply_transform(held_obj_pose_transformed)
                center_point = copied_mesh.center_mass

                if "table" in support_obj_name:
                    if not (min_x - 0.2 <= center_point[0] <= min_x):
                        continue
                    if not (min_y - 0.1 <= center_point[1] <= max_y + 0.1):
                        continue
                elif "shelf_9" in support_obj_name:
                    center_point = copied_mesh.bounds[0] + (copied_mesh.bounds[1] - copied_mesh.bounds[0])/2
                    if not (min_x + 0.05 <= center_point[0] <= max_x - 0.05):
                        continue
                    if not (min_y - 0.5 <= center_point[1] <= max_y - 0.5):
                        continue
                else:
                    if not (min_x <= center_point[0] <= max_x):
                        continue
                    if not (min_y <= center_point[1] <= max_y):
                        continue
                
                if eef_pose is not None:
                    T_obj_pose_and_obj_pose_transformed = np.dot(held_obj_pose, np.linalg.inv(held_obj_pose_rotated))
                    eef_pose_transformed = self._get_eef_pose_transformed(
                        T_obj_pose_and_obj_pose_transformed, eef_pose, support_obj_point, held_obj_point_transformed)
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
    def _check_stability(copied_scene:Scene, held_obj_name:str, com:np.ndarray) -> bool:
        if copied_scene.logical_state.on in list(copied_scene.logical_states[held_obj_name].keys()):
            support_obj = copied_scene.logical_states[held_obj_name][copied_scene.logical_state.on]
            support_obj_mesh:Trimesh = deepcopy(support_obj.gparam)
            support_obj_mesh.apply_transform(support_obj.h_mat)

            closest_points, _, _ = proximity.closest_point(support_obj_mesh, [com])
            closest_point = closest_points[0]
            norm_vector = m_utils.normalize(closest_point - com)

            safe_com = norm_vector[2] == -1.0
            if safe_com:
                return PlaceAction._check_stability(copied_scene, support_obj.name, com)
            else:
                # logger.warning("Not satisfied stability")
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
        place_obj_mesh:Trimesh = deepcopy(place_obj.gparam)
        place_obj_mesh.apply_transform(place_obj.h_mat)

        locations, _, _ = place_obj_mesh.ray.intersects_location(
                    ray_origins=[held_obj_mesh.center_mass],
                    ray_directions=[[0, 0, -1]])
        if len(locations) != 0:
            return True
        # logger.warning("Not found support point")
        return False