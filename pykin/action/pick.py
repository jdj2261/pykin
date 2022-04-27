from cmath import pi
import numpy as np
from enum import Enum, auto
from copy import deepcopy

import pykin.utils.action_utils as a_utils
from pykin.action.activity import ActivityBase
from pykin.scene.scene import Scene, SceneManager

class GraspPose(Enum):
    """
    Grasp Status Enum class
    """
    PRE_GRASP_POSE = auto()
    GRASP_POSE = auto()
    POST_GRASP = auto()

class PickAction(ActivityBase):
    def __init__(
        self, 
        scene_mngr,
        n_contacts=10,
        n_directions=10,
        limit_angle_for_force_closure=0.05,
    ):
        super().__init__(scene_mngr)
        self.n_contacts = n_contacts
        self.n_directions = n_directions
        self.limit_angle = limit_angle_for_force_closure
        self.filter_logical_states = [ scene_mngr.scene.state.support, 
                                       scene_mngr.scene.state.static]

    def get_possible_actions(self, scene:Scene=None, level=0):
        if not 0 <= level <= 2:
            raise ValueError("Check level number!!")

        if scene is None:
            scene = self.scene_mngr.scene

        self.scene_mngr = self.scene_mngr.copy_scene(self.scene_mngr)
        self.scene_mngr.scene = deepcopy(scene)
        self.scene_mngr.show_logical_states()
        
        for obj in self.scene_mngr.scene.objs:
            if not any(logical_state in self.scene_mngr.scene.logical_states[obj] for logical_state in self.filter_logical_states):
                grasp_poses = list(self.get_grasp_poses(obj_name=obj))
                action = self.get_action(obj, grasp_poses)
                if level == 0:
                    yield action, None, None
                elif level <= 2:
                    grasp_poses_for_only_gripper = list(self.get_grasp_poses_for_only_gripper(grasp_poses))
                    action_level_1 = self.get_action(obj, grasp_poses_for_only_gripper)
                    if level == 1:
                        yield action, action_level_1, None
                    else:
                        goal_grasp_poses = list(self.get_grasp_poses_for_robot(grasp_poses_for_only_gripper))
                        action_level_2 = self.get_action(obj, goal_grasp_poses)
                        yield action, action_level_1, action_level_2

    def get_action(self, obj_name, poses):
        action = {}
        action[self.action_info.ACTION] = "pick"
        action[self.action_info.OBJ_NAME] = obj_name
        action[self.action_info.GRASP_POSES] = poses
        return action

    def get_possible_transitions(self, scene:Scene=None, action:dict={}):        
        if not action:
            ValueError("Not found any action!!")

        pick_obj = action[self.action_info.OBJ_NAME]

        for grasp_pose in action[self.action_info.GRASP_POSES]:
            next_scene = deepcopy(scene)
            supporting_obj = next_scene.logical_states[pick_obj].get(next_scene.state.on)
            next_scene.logical_states.get(supporting_obj.name).get(next_scene.state.support).remove(next_scene.objs[pick_obj])
            
            # Clear logical_state of pick obj
            next_scene.logical_states[pick_obj].clear()
            next_scene.robot.gripper.set_gripper_pose(grasp_pose)
            
            # Add logical_state of pick obj : {'held' : True}
            next_scene.logical_states[self.scene_mngr.gripper_name][next_scene.state.holding] = next_scene.objs[pick_obj]
            next_scene.update_logical_states()
            yield next_scene
            
    # Not consider collision
    def get_grasp_poses(self, obj_name):
        if self.scene_mngr.scene.robot.has_gripper is None:
            raise ValueError("Robot doesn't have a gripper")

        gripper = self.scene_mngr.scene.robot.gripper
        tcp_poses = self.get_tcp_poses(obj_name)
        for tcp_pose in tcp_poses:
            grasp_pose = gripper.compute_eef_pose_from_tcp_pose(tcp_pose)
            yield grasp_pose

    # for level wise - 1 (Consider gripper collision)
    def get_grasp_poses_for_only_gripper(self, grasp_poses):
        if not grasp_poses:
            raise ValueError("Not found grasp poses!")

        for grasp_pose in grasp_poses:
            self.scene_mngr.set_gripper_pose(grasp_pose)
            if not self._collide(is_only_gripper=True):
                yield grasp_pose

    # for level wise - 2 (Consider IK and collision)
    def get_grasp_poses_for_robot(self, grasp_poses_for_only_grpper):
        if self.scene_mngr.scene.robot is None:
            raise ValueError("Robot needs to be added first")

        grasp_poses = grasp_poses_for_only_grpper
        if not grasp_poses:
            return

        for grasp_pose in grasp_poses:
            thetas = self.scene_mngr.compute_ik(pose=grasp_pose, max_iter=100)
            self.scene_mngr.set_robot_eef_pose(thetas)
            grasp_pose_from_ik = self.scene_mngr.get_robot_eef_pose()

            if self._solve_ik(grasp_pose, grasp_pose_from_ik) and not self._collide(is_only_gripper=False):
                yield grasp_pose

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