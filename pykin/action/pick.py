
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

import pykin.utils.action_utils as a_utils
from pykin.action.activity import ActivityBase
from pykin.scene.scene import Scene
from pykin.utils.action_utils import get_relative_transform

@dataclass
class GraspName:
    """
    Grasp Status Enum class
    """
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"
    POST_GRASP = "post_grasp"

class PickAction(ActivityBase):
    def __init__(
        self, 
        scene_mngr,
        n_contacts=10,
        n_directions=10,
        limit_angle_for_force_closure=0.05,
        retreat_distance=0.1
    ):
        super().__init__(scene_mngr, retreat_distance)
        self.grasp_name = GraspName
        self.n_contacts = n_contacts
        self.n_directions = n_directions
        self.limit_angle = limit_angle_for_force_closure
        self.filter_logical_states = [ scene_mngr.scene.state.support, 
                                       scene_mngr.scene.state.static]

    # Expand action to tree
    def get_possible_actions_level_1(self, scene:Scene=None) -> dict:
        if scene is None:
            scene = self.scene_mngr.scene

        self.scene_mngr.scene = deepcopy(scene)
        
        for obj in self.scene_mngr.scene.objs:
            if obj == self.scene_mngr.scene.pick_obj:
                continue
            
            if not any(logical_state in self.scene_mngr.scene.logical_states[obj] for logical_state in self.filter_logical_states):
                grasp_poses = list(self.get_all_grasp_poses(obj_name=obj))
                grasp_poses_for_only_gripper = list(self.get_all_grasp_poses_for_only_gripper(grasp_poses))
                action_level_1 = self.get_action(obj, grasp_poses_for_only_gripper)
                yield action_level_1

    # Not Expand, only check possible action using ik
    def get_possible_ik_solve_level_2(self, scene:Scene=None, grasp_pose:dict={}) -> bool:
        if scene is None:
            scene = self.scene_mngr.scene
        self.scene_mngr.scene = deepcopy(scene)
        
        ik_solve = self.compute_ik_solve_for_robot(grasp_pose)
        return ik_solve
 
    def get_action(self, obj_name, all_poses):
        action = {}
        action[self.action_info.ACTION] = "pick"
        action[self.action_info.PICK_OBJ_NAME] = obj_name
        action[self.action_info.GRASP_POSES] = all_poses
        return action

    def get_possible_transitions(self, scene:Scene=None, action:dict={}):        
        if not action:
            ValueError("Not found any action!!")

        pick_obj = action[self.action_info.PICK_OBJ_NAME]

        for grasp_pose in action[self.action_info.GRASP_POSES]:
            next_scene = deepcopy(scene)
            
            supporting_obj = next_scene.logical_states[pick_obj].get(next_scene.state.on)
            next_scene.place_obj = supporting_obj.name
            next_scene.logical_states.get(supporting_obj.name).get(next_scene.state.support).remove(next_scene.objs[pick_obj])
            
            # Clear logical_state of pick obj
            next_scene.logical_states[pick_obj].clear()

            # Gripper Move to grasp pose
            next_scene.robot.gripper.set_gripper_pose(grasp_pose[self.grasp_name.GRASP])
            gripper_pose = deepcopy(next_scene.robot.gripper.get_gripper_pose())
            transform_bet_gripper_n_obj = get_relative_transform(gripper_pose, next_scene.objs[pick_obj].h_mat)
            
            # Attach Object
            next_scene.robot.gripper.attached_obj_name = pick_obj
            next_scene.robot.gripper.grasp_pose = gripper_pose
            next_scene.robot.gripper.transform_bet_gripper_n_obj = transform_bet_gripper_n_obj
            next_scene.robot.gripper.pick_obj_pose = deepcopy(next_scene.objs[pick_obj].h_mat)
            # self.scene_mngr.obj_collision_mngr.set_transform(pick_obj, deepcopy(next_scene.objs[pick_obj].h_mat))
            # self.scene_mngr.obj_collision_mngr.show_collision_info("Object")
            # Gripper Move to default pose
            next_scene.robot.gripper.set_gripper_pose(next_scene.robot.get_gripper_init_pose())
            
            # pick object Move to default pose with gripper
            next_scene.objs[pick_obj].h_mat = np.dot(next_scene.robot.gripper.get_gripper_pose(), transform_bet_gripper_n_obj)
            
            # Add logical_state of pick obj : {'held' : True}
            next_scene.logical_states[self.scene_mngr.gripper_name][next_scene.state.holding] = next_scene.objs[pick_obj]
            next_scene.update_logical_states()
            yield next_scene
            
    # Not consider collision
    def get_all_grasp_poses(self, obj_name:str) -> dict:
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        gripper = self.scene_mngr.scene.robot.gripper
        tcp_poses = self.get_tcp_poses(obj_name)
        
        for tcp_pose in tcp_poses:
            grasp_pose = {}
            grasp_pose[self.grasp_name.GRASP] = gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
            grasp_pose[self.grasp_name.PRE_GRASP] = self.get_pre_grasp_pose(grasp_pose[self.grasp_name.GRASP])
            grasp_pose[self.grasp_name.POST_GRASP] = self.get_post_grasp_pose(grasp_pose[self.grasp_name.GRASP])
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
    def get_all_grasp_poses_for_only_gripper(self, grasp_poses):
        if not grasp_poses:
            raise ValueError("Not found grasp poses!")

        for all_grasp_pose in grasp_poses:
            for name, pose in all_grasp_pose.items():
                is_collision = False
                if name == self.grasp_name.GRASP:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.grasp_name.PRE_GRASP:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
                if name == self.grasp_name.POST_GRASP:
                    self.scene_mngr.set_gripper_pose(pose)
                    if self._collide(is_only_gripper=True):
                        is_collision = True
                        break
            
            if not is_collision:
                yield all_grasp_pose
            
    def compute_ik_solve_for_robot(self, grasp_pose:dict):
        ik_sovle = {}

        for name, pose in grasp_pose.items():
            if name == self.grasp_name.GRASP:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_sovle[self.grasp_name.GRASP] = thetas
            if name == self.grasp_name.PRE_GRASP:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                pre_grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, pre_grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_sovle[self.grasp_name.PRE_GRASP] = thetas
            if name == self.grasp_name.POST_GRASP:
                thetas = self.scene_mngr.compute_ik(pose=pose, max_iter=100)
                self.scene_mngr.set_robot_eef_pose(thetas)
                post_grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()
                if self._solve_ik(pose, post_grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                    ik_sovle[self.grasp_name.POST_GRASP] = thetas
        
        if len(ik_sovle) == 3:
            return ik_sovle
        return None

    def get_contact_points(self, obj_name):
        copied_mesh = deepcopy(self.scene_mngr.scene.objs[obj_name].gparam)
        copied_mesh.apply_transform(self.scene_mngr.scene.objs[obj_name].h_mat)
        
        cnt = 0
        while cnt < self.n_contacts:
            surface_points, normals = self.get_surface_points_from_mesh(copied_mesh, 2)
            if self._is_force_closure(surface_points, normals, self.limit_angle):
                cnt += 1
                yield surface_points

    def _is_force_closure(self, points, normals, limit_angle):
        vectorA = points[0]
        vectorB = points[1]

        normalA = -normals[0]
        normalB = -normals[1]

        vectorAB = vectorB - vectorA
        distance = np.linalg.norm(vectorAB)

        unit_vectorAB = a_utils.normalize(vectorAB)
        angle_A2AB = np.arccos(normalA.dot(unit_vectorAB))

        unit_vectorBA = -1 * unit_vectorAB
        angle_B2AB = np.arccos(normalB.dot(unit_vectorBA))

        if distance > self.scene_mngr.scene.robot.gripper.max_width:
            return False

        if angle_A2AB > limit_angle or angle_B2AB > limit_angle:
            return False    
        return True

    def get_tcp_poses(self, obj_name):
        contact_points = list(self.get_contact_points(obj_name))
        if not contact_points:
            raise ValueError("Cannot get tcp poses!!")
        
        for contact_point in contact_points:
            p1, p2 = contact_point
            center_point = (p1 + p2) /2
            line = p2 - p1

            for _, grasp_dir in enumerate(a_utils.get_grasp_directions(line, self.n_directions)):
                y = a_utils.normalize(line)
                z = grasp_dir
                x = np.cross(y, z)

                tcp_pose = np.eye(4)
                tcp_pose[:3,0] = x
                tcp_pose[:3,1] = y
                tcp_pose[:3,2] = z
                tcp_pose[:3,3] = center_point

                yield tcp_pose