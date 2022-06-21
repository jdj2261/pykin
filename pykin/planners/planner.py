import numpy as np
from abc import abstractclassmethod, ABCMeta
from dataclasses import dataclass
from pykin.scene.scene_manager import SceneManager

from pykin.utils.log_utils import create_logger
from pykin.utils.error_utils import CollisionError, NotFoundError

logger = create_logger('Cartesian Planner', "debug",)

@dataclass
class NodeData:
    COST = 'cost'
    POINT = 'point'

class Planner(NodeData, metaclass=ABCMeta):
    """
    Base Planner class 

    Args:
        dimension(int): robot arm's dof
    """
    def __init__(
        self,
        dimension
    ):
        self._dimension = dimension
        self._cur_qpos = None
        self._goal_pose = None
        self._max_iter = None
        self._scene_mngr:SceneManager = None
        self.joint_path = None
        self.arm = None

    def __repr__(self) -> str:
        return 'pykin.planners.planner.{}()'.format(type(self).__name__)

    @abstractclassmethod
    def run(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_joint_path(self):
        """
        write planner algorithm you want 
        """
        raise NotImplementedError

    def get_target_eef_poses(self):
        if not self.joint_path:
            ValueError("Cannot get target eef poses, because the joint path has not been optained")
        
        eef_poses = []
        for step, joint in enumerate(self.joint_path):
            fk = self._scene_mngr.scene.robot.forward_kin(joint)
            eef_poses.append(fk[self._scene_mngr.scene.robot.eef_name].pos)
        return eef_poses

    @abstractclassmethod
    def _get_linear_path(self, init_pose, goal_pose):
        """
        Base Planner class 

        Args:
            init_pose (np.array): init robots' eef pose
            goal_pose (np.array): goal robots' eef pose        
        """
        raise NotImplementedError

    @staticmethod
    def _convert_numpy_type(data):
        """
        Convert input data type to numpy type

        Args:
            data (sequence of float): input data

        Returns:
            np_data (np.array)
        """
        np_data = np.array(data)
        if not isinstance(np_data, (np.ndarray)):
            print(np_data.size)
            if np_data.size == 0:
                raise NotFoundError("Make sure set current or goal joints..")
        return np_data

    def _setup_q_limits(self):
        """
        Setup joint limits (lower and upper)
        """
        if self.arm is not None:
            self.q_limits_lower = self._scene_mngr.scene.robot.joint_limits_lower[self.arm]
            self.q_limits_upper = self._scene_mngr.scene.robot.joint_limits_upper[self.arm]
        else:
            self.q_limits_lower = self._scene_mngr.scene.robot.joint_limits_lower
            self.q_limits_upper = self._scene_mngr.scene.robot.joint_limits_upper

    def _check_q_in_limits(self, q_in):
        """
        check q_in within joint limits
        If q_in is in joint limits, return True
        otherwise, return False

        Returns:
            bool (True or False)
        """
        return np.all([q_in >= self.q_limits_lower, q_in <= self.q_limits_upper])

    def _check_robot_col_mngr(self):
        if self._scene_mngr.robot_collision_mngr is None:
            return False
        
        if self._scene_mngr.collide_self_robot():
            raise CollisionError("Conflict confirmed. Check the joint settings again")
        
        return True

    def _setup_eef_name(self):
        """
        Setup end-effector name
        """
        if self.arm is not None:
            self.eef_name = self._scene_mngr.scene.robot.eef_name[self.arm]

    def _collide(
        self, 
        new_q, 
        only_robot=False,
        visible_name=False
    ):
        """
        Check collision free between robot and objects

        Args:
            new_q (np.array): new joint angles
            visible_name (bool): If it's true, the result of the collision and the name will come out. 
                                 Otherwise, only the collision results will come out.

        Returns:
            result (bool): If collision free, return True
            names (set of 2-tup): The set of pairwise collisions. 
        """
        is_attached = self._scene_mngr.is_attached
        if only_robot:
            is_attached = False

        if self._scene_mngr.robot_collision_mngr is None:
            return False

        fk = self._get_fk(new_q)

        for link, transform in fk.items():
            if link in self._scene_mngr.robot_collision_mngr._objs:
                if self._scene_mngr.robot_collision_mngr.geom == "visual":
                    h_mat = np.dot(transform.h_mat, self._scene_mngr.scene.robot.links[link].visual.offset.h_mat)
                else:
                    h_mat = np.dot(transform.h_mat, self._scene_mngr.scene.robot.links[link].collision.offset.h_mat)
                self._scene_mngr.robot_collision_mngr.set_transform(name=link, h_mat=h_mat)
        
        if is_attached:
            gripper_pose = fk[self._scene_mngr.scene.robot.eef_name].h_mat
            h_mat = np.dot(gripper_pose, self._scene_mngr._transform_bet_gripper_n_obj)
            self._scene_mngr.robot_collision_mngr.set_transform(name=self._scene_mngr.attached_obj_name, h_mat=h_mat)

        is_self_collision = self._scene_mngr.robot_collision_mngr.in_collision_internal(return_names=False)

        if visible_name:
            is_object_collision, col_name = self._scene_mngr.robot_collision_mngr.in_collision_other(
                other_manager=self._scene_mngr.obj_collision_mngr, return_names=visible_name)  
            if is_self_collision or is_object_collision:
                return True, col_name
            return False, col_name
            
        is_object_collision = self._scene_mngr.robot_collision_mngr.in_collision_other(
            other_manager=self._scene_mngr.obj_collision_mngr, return_names=False)  
        if is_self_collision or is_object_collision:
            return True
        return False

    def _get_fk(self, q_in):
        """
        Get forward kinematics corresponding to q_in

        Args:
            q_in (np.array): joint angles

        Returns:
            fk (OrderedDict)
        """
        if self._scene_mngr.scene.robot.robot_name == "sawyer":
            q_in = np.concatenate((np.zeros(1), q_in))

        if self.arm is not None:
            fk = self._scene_mngr.scene.robot.forward_kin(q_in, self._scene_mngr.scene.robot.desired_frames[self.arm])
        else:
            fk = self._scene_mngr.scene.robot.forward_kin(q_in)
        return fk

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, dimesion):
        self._dimension = dimesion

    @property
    def cur_qpos(self):
        return self._cur_qpos

    @cur_qpos.setter
    def cur_qpos(self, cur_qpos):
        self._cur_qpos = cur_qpos

    @property
    def goal_pose(self):
        return self._goal_pose

    @goal_pose.setter
    def goal_pose(self, goal_pose):
        self._goal_pose = goal_pose