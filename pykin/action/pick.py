import numpy as np
from collections import OrderedDict
from copy import deepcopy

import pykin.utils.mesh_utils as m_utils
from pykin.action.activity import ActivityBase
from pykin.scene.scene import Scene
from pykin.utils.mesh_utils import get_relative_transform
from pykin.utils.transform_utils import get_rpy_from_matrix

class PickAction(ActivityBase):
    def __init__(
        self, 
        scene_mngr,
        n_contacts=10,
        n_directions=10,
        limit_angle_for_force_closure=0.2,
        retreat_distance=0.1
    ):
        super().__init__(scene_mngr, retreat_distance)
        self.n_contacts = n_contacts
        self.n_directions = n_directions
        self.limit_angle = limit_angle_for_force_closure
        self.filter_logical_states = [ scene_mngr.scene.logical_state.support, 
                                       scene_mngr.scene.logical_state.static]

    # Expand action to tree
    def get_possible_actions_level_1(self, scene:Scene=None) -> dict:
        self.deepcopy_scene(scene)
        
        for obj_name in self.scene_mngr.scene.objs:
            if obj_name == self.scene_mngr.scene.pick_obj_name:
                continue

            if self.scene_mngr.scene.bench_num == 4 and "_6" not in obj_name:
                continue

            if not any(logical_state in self.scene_mngr.scene.logical_states[obj_name] for logical_state in self.filter_logical_states):
                print(f"pick : {obj_name}")
                action_level_1 = self.get_action_level_1_for_single_object(obj_name=obj_name)
                if not action_level_1[self.info.GRASP_POSES]:
                    continue
                yield action_level_1
    
    def get_action_level_1_for_single_object(self, scene=None, obj_name:str=None) -> dict:
        if scene is not None:
            self.deepcopy_scene(scene)

        grasp_poses = list(self.get_all_grasp_poses(obj_name=obj_name))
        grasp_poses.extend(list(self.get_grasp_pose_from_heuristic(obj_name)))
        grasp_poses_not_collision = list(self.get_all_grasp_poses_not_collision(grasp_poses))

        action_level_1 = self.get_action(obj_name, grasp_poses_not_collision)
        return action_level_1

    def get_grasp_pose_from_heuristic(self, obj_name, dis_z=0.01):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)
        
        if "box" in obj_name or "hanoi_disk" in obj_name:
            obj_pose = np.eye(4)
            center_point = copied_mesh.center_mass
            obj_pose[:3, :3] = self.scene_mngr.scene.objs[obj_name].h_mat[:3, :3]
            
            obj_pose[1,:3] = abs(obj_pose[1,:3])
            obj_pose[0,:3] = np.cross(obj_pose[1,:3], obj_pose[2,:3])
            obj_pose[:3, 3] = center_point + [0, 0, dis_z]
        
            for theta in np.linspace(np.pi, np.pi-np.pi/24, 1):
            # for theta in np.linspace(0, np.pi*2, 50):
                tcp_pose = np.eye(4)
                tcp_pose[:3,0] = [np.cos(theta), 0, np.sin(theta)]
                tcp_pose[:3,1] = [0, 1, 0]
                tcp_pose[:3,2] = [-np.sin(theta), 0, np.cos(theta)]
                # tcp_pose[:3, 3] = center_point + [0, 0, dis_z]
                tcp_pose = np.dot(obj_pose, tcp_pose)

                grasp_pose = {}
                grasp_pose[self.move_data.MOVE_grasp] = self.scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
                grasp_pose[self.move_data.MOVE_pre_grasp] = self.get_pre_grasp_pose(grasp_pose[self.move_data.MOVE_grasp])
                grasp_pose[self.move_data.MOVE_post_grasp] = self.get_post_grasp_pose(grasp_pose[self.move_data.MOVE_grasp])        
            
                yield grasp_pose

        if "bottle" in obj_name:
            center_point = copied_mesh.bounds[0] + (copied_mesh.bounds[1] - copied_mesh.bounds[0])/2
            for theta in np.linspace(-np.pi+np.pi/(2.2), -np.pi/(2.2), 3):
                tcp_pose = np.eye(4)
                tcp_pose[:3,0] = [np.cos(theta), 0, np.sin(theta)]
                tcp_pose[:3,1] = [0, 1, 0]
                tcp_pose[:3,2] = [-np.sin(theta), 0, np.cos(theta)]
                tcp_pose[:3,3] = center_point + [0, 0, 0.005]

                grasp_pose = {}
                grasp_pose[self.move_data.MOVE_grasp] = self.scene_mngr.scene.robot.gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
                grasp_pose[self.move_data.MOVE_pre_grasp] = self.get_pre_grasp_pose(grasp_pose[self.move_data.MOVE_grasp])
                grasp_pose[self.move_data.MOVE_post_grasp] = self.get_post_grasp_pose(grasp_pose[self.move_data.MOVE_grasp])        
                yield grasp_pose
    # Not Expand, only check possible action using ik

    def get_possible_ik_solve_level_2(self, scene:Scene=None, grasp_poses:dict={}) -> bool:
        self.deepcopy_scene(scene)
        
        ik_solve, grasp_poses_filtered = self.compute_ik_solve_for_robot(grasp_poses)
        return ik_solve, grasp_poses_filtered
 
    def get_possible_joint_path_level_3(self, scene:Scene=None, grasp_poses:dict={}, init_thetas=None):
        self.deepcopy_scene(scene)
        
        pick_obj = self.scene_mngr.scene.robot.gripper.attached_obj_name
        self.scene_mngr.scene.objs[pick_obj].h_mat = self.scene_mngr.scene.robot.gripper.pick_obj_pose
        for obj_name in self.scene_mngr.scene.objs:
            obj_pose = self.scene_mngr.scene.objs[obj_name].h_mat
            if obj_name == pick_obj:
                obj_pose = self.scene_mngr.scene.robot.gripper.pick_obj_pose
            self.scene_mngr.obj_collision_mngr.set_transform(obj_name, obj_pose)

        result_all_joint_path = []
        result_joint_path = OrderedDict()
        default_joint_path = []

        default_thetas = init_thetas
        if init_thetas is None:
            default_thetas = self.scene_mngr.scene.robot.init_qpos
            
        pre_grasp_pose = grasp_poses[self.move_data.MOVE_pre_grasp]
        grasp_pose = grasp_poses[self.move_data.MOVE_grasp]
        post_grasp_pose = grasp_poses[self.move_data.MOVE_post_grasp]
        success_joint_path = True

        # default pose -> pre_grasp_pose (rrt)
        pre_grasp_joint_path = self.get_rrt_star_path(default_thetas, pre_grasp_pose)
        if pre_grasp_joint_path:
            # pre_grasp_pose -> grasp_pose (cartesian)
            grasp_joint_path = self.get_cartesian_path(pre_grasp_joint_path[-1], grasp_pose)
            if grasp_joint_path:
                # grasp_pose -> post_grasp_pose (cartesian)
                self.scene_mngr.set_robot_eef_pose(grasp_joint_path[-1])
                self.scene_mngr.attach_object_on_gripper(self.scene_mngr.scene.robot.gripper.attached_obj_name)
                post_grasp_joint_path = self.get_cartesian_path(grasp_joint_path[-1], post_grasp_pose)
                if post_grasp_joint_path:
                    # post_grasp_pose -> default pose (rrt)
                    default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)["right_gripper"].h_mat
                    default_joint_path = self.get_rrt_star_path(post_grasp_joint_path[-1], default_pose)
                else:
                    success_joint_path = False
                self.scene_mngr.detach_object_from_gripper()
                self.scene_mngr.add_object(
                    self.scene_mngr.scene.robot.gripper.attached_obj_name,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gtype,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].gparam,
                    self.scene_mngr.scene.robot.gripper.pick_obj_pose,
                    self.scene_mngr.init_objects[self.scene_mngr.scene.robot.gripper.attached_obj_name].color)
            else:
                success_joint_path = False
        else:
            success_joint_path = False

        if not success_joint_path:
            return result_all_joint_path

        if default_joint_path:
            result_joint_path.update({self.move_data.MOVE_pre_grasp: pre_grasp_joint_path})
            result_joint_path.update({self.move_data.MOVE_grasp: grasp_joint_path})
            result_joint_path.update({self.move_data.MOVE_post_grasp: post_grasp_joint_path})
            result_joint_path.update({self.move_data.MOVE_default_grasp: default_joint_path})
            result_all_joint_path.append(result_joint_path)
            return result_all_joint_path

    def get_action(self, obj_name, all_poses):
        action = {}
        action[self.info.TYPE] = "pick"
        action[self.info.PICK_OBJ_NAME] = obj_name
        action[self.info.GRASP_POSES] = all_poses
        return action

    def get_possible_transitions(self, scene:Scene=None, action:dict={}):        
        if not action:
            ValueError("Not found any action!!")

        pick_obj = action[self.info.PICK_OBJ_NAME]

        for grasp_poses in action[self.info.GRASP_POSES]:
            next_scene = deepcopy(scene)
            
            ## Change transition
            next_scene.grasp_poses = grasp_poses
            next_scene.robot.gripper.grasp_pose = grasp_poses[self.move_data.MOVE_grasp]
            
            # Gripper Move to grasp pose
            next_scene.robot.gripper.set_gripper_pose(grasp_poses[self.move_data.MOVE_grasp])
            
            # Get transform between gripper and pick object
            gripper_pose = deepcopy(next_scene.robot.gripper.get_gripper_pose())
            transform_bet_gripper_n_obj = get_relative_transform(gripper_pose, next_scene.objs[pick_obj].h_mat)
            
            # Attach Object to gripper
            next_scene.robot.gripper.attached_obj_name = pick_obj
            next_scene.robot.gripper.pick_obj_pose = deepcopy(next_scene.objs[pick_obj].h_mat)
            next_scene.robot.gripper.transform_bet_gripper_n_obj = transform_bet_gripper_n_obj

            # Move a gripper to default pose
            default_thetas = self.scene_mngr.scene.robot.init_qpos
            default_pose = self.scene_mngr.scene.robot.forward_kin(default_thetas)["right_gripper"].h_mat
            next_scene.robot.gripper.set_gripper_pose(default_pose)
            
            # Move pick object to default pose
            next_scene.objs[pick_obj].h_mat = np.dot(next_scene.robot.gripper.get_gripper_pose(), transform_bet_gripper_n_obj)
            next_scene.pick_obj_default_pose = deepcopy(next_scene.objs[pick_obj].h_mat)
            
            ## Change Logical State
            # Remove pick obj in logical state of support obj
            supporting_obj = next_scene.logical_states[pick_obj].get(next_scene.logical_state.on)
            next_scene.place_obj_name = supporting_obj.name
            next_scene.logical_states.get(supporting_obj.name).get(next_scene.logical_state.support).remove(next_scene.objs[pick_obj])
            
            # Clear logical_state of pick obj
            next_scene.logical_states[pick_obj].clear()
            
            # Add logical_state of pick obj : {'held' : True}
            next_scene.logical_states[self.scene_mngr.gripper_name][next_scene.logical_state.holding] = next_scene.objs[pick_obj]
            next_scene.update_logical_states()
            yield next_scene
            
    # Not consider collision
    def get_all_grasp_poses(self, obj_name:str) -> dict:
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        gripper = self.scene_mngr.scene.robot.gripper
        for tcp_pose in self.get_tcp_poses(obj_name):
            grasp_pose = {}
            grasp_pose[self.move_data.MOVE_grasp] = gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
            grasp_pose[self.move_data.MOVE_pre_grasp] = self.get_pre_grasp_pose(grasp_pose[self.move_data.MOVE_grasp])
            grasp_pose[self.move_data.MOVE_post_grasp] = self.get_post_grasp_pose(grasp_pose[self.move_data.MOVE_grasp])
            yield grasp_pose

    def get_pre_grasp_pose(self, grasp_pose):
        pre_grasp_pose = np.eye(4)
        pre_grasp_pose[:3, :3] = grasp_pose[:3, :3]
        pre_grasp_pose[:3, 3] = grasp_pose[:3, 3] - self.retreat_distance * grasp_pose[:3,2]    
        return pre_grasp_pose

    def get_post_grasp_pose(self, grasp_pose):
        post_grasp_pose = np.eye(4)
        post_grasp_pose[:3, :3] = grasp_pose[:3, :3] 
        post_grasp_pose[:3, 3] = grasp_pose[:3, 3] + np.array([0, 0, self.retreat_distance])  
        return post_grasp_pose

    # for level wise - 1 (Consider gripper collision)
    def get_all_grasp_poses_not_collision(self, grasp_poses):
        if not grasp_poses:
            raise ValueError("Not found grasp poses!")

        for all_grasp_pose in grasp_poses:
            for name, pose in all_grasp_pose.items():
                is_collision = False
                if name == self.move_data.MOVE_grasp:
                    self.scene_mngr.set_gripper_pose(pose)
                    for name in self.scene_mngr.scene.objs:
                        self.scene_mngr.obj_collision_mngr.set_transform(name, self.scene_mngr.scene.objs[name].h_mat)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.move_data.MOVE_pre_grasp:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.move_data.MOVE_post_grasp:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
            
            if not is_collision:
                yield all_grasp_pose
            
    def compute_ik_solve_for_robot(self, grasp_pose:dict):
        ik_solve = {}
        grasp_pose_for_ik = {}

        for name, pose in grasp_pose.items():
            if name == self.move_data.MOVE_grasp:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_solve[name] = thetas
                    grasp_pose_for_ik[name] = pose
            if name == self.move_data.MOVE_pre_grasp:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                pre_grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, pre_grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_solve[name] = thetas
                    grasp_pose_for_ik[name] = pose
            if name == self.move_data.MOVE_post_grasp:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                post_grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, post_grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_solve[name] = thetas
                    grasp_pose_for_ik[name] = pose
        
        if len(ik_solve) == 3:
            return ik_solve, grasp_pose_for_ik
        return None, None

    def get_contact_points(self, obj_name):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)

        center_point = copied_mesh.center_mass

        len_x = abs(center_point[0] - copied_mesh.bounds[0][0])
        len_y = abs(center_point[1] - copied_mesh.bounds[0][1])
        len_z = abs(center_point[2] - copied_mesh.bounds[0][2])

        weights = self._get_weights_for_held_obj(copied_mesh)

        cnt = 0
        margin = 0.8
        surface_point_list = []
        while cnt < self.n_contacts:
            surface_points, normals = self.get_surface_points_from_mesh(copied_mesh, 2, weights)
            is_success = False
            if self._is_force_closure(surface_points, normals, self.limit_angle):
                
                if center_point[0] - len_x * margin <= surface_points[0][0] <= center_point[0] + len_x * margin:
                    if center_point[1] - len_y * margin <= surface_points[0][1] <= center_point[1] + len_y * margin:
                        is_success = True
                        
                if center_point[2] - len_z * margin <= surface_points[0][2] <= center_point[2] + len_z * margin:
                    if center_point[1] - len_y * margin <= surface_points[0][1] <= center_point[1] + len_y * margin:
                        is_success = True

                if center_point[2] - len_z * margin <= surface_points[0][2] <= center_point[2] + len_z * margin:
                    if center_point[0] - len_x * margin <= surface_points[0][0] <= center_point[0] + len_x * margin:
                        is_success = True
                
                if is_success:
                    cnt += 1
                    surface_point_list.append(surface_points)
        
        return surface_point_list
                
    @staticmethod
    def _get_weights_for_held_obj(obj_mesh):
        # heuristic
        weights = np.zeros(len(obj_mesh.faces))
        for idx, vertex in enumerate(obj_mesh.vertices[obj_mesh.faces]):
            weights[idx]=0.05
            if np.all(vertex[:,2] <= obj_mesh.bounds[0][2] * 1.02):                
                weights[idx] = 0.95
        return weights

    def _is_force_closure(self, points, normals, limit_angle):
        vectorA = points[0]
        vectorB = points[1]

        normalA = -normals[0]
        normalB = -normals[1]

        vectorAB = vectorB - vectorA
        distance = np.linalg.norm(vectorAB)

        unit_vectorAB = m_utils.normalize(vectorAB)
        angle_A2AB = np.arccos(normalA.dot(unit_vectorAB))

        unit_vectorBA = -1 * unit_vectorAB
        angle_B2AB = np.arccos(normalB.dot(unit_vectorBA))

        if distance > self.scene_mngr.scene.robot.gripper.max_width:
            return False

        if angle_A2AB > limit_angle or angle_B2AB > limit_angle:
            return False    
        return True

    def get_tcp_poses(self, obj_name):
        contact_points = self.get_contact_points(obj_name)
        
        for contact_point in contact_points:
            p1, p2 = contact_point
            center_point = (p1 + p2) /2
            line = p2 - p1

            for _, grasp_dir in enumerate(m_utils.get_grasp_directions(line, self.n_directions)):
                y = m_utils.normalize(line)
                z = grasp_dir
                x = np.cross(y, z)

                tcp_pose = np.eye(4)
                tcp_pose[:3,0] = x
                tcp_pose[:3,1] = y
                tcp_pose[:3,2] = z
                tcp_pose[:3,3] = center_point

                yield tcp_pose