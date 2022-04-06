import numpy as np
from abc import abstractclassmethod, ABCMeta
from dataclasses import dataclass

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
        robot (SingleArm or Bimanual): manipulator type
        dimension(int): robot arm's dof
    """
    def __init__(
        self,
        robot,
        dimension
    ):
        self.robot = robot
        self._dimension = dimension
        
        self.robot_col_mngr = None
        self.object_col_mngr = None
        self.is_attached = None
        self.obj_info = None
        self.T_between_grippper_and_obj = None
        self.result_object_pose = None

    def __repr__(self) -> str:
        return 'pykin.planners.planner.{}()'.format(type(self).__name__)

    def attach_object_on_robot(self):
        """
        Attach object collision on robot collision
        """
        self.robot_col_mngr.add_object(
            self.obj_info["name"], 
            gtype=self.obj_info["gtype"], 
            gparam=self.obj_info["gparam"], 
            h_mat=self.obj_info["transform"])
        self.object_col_mngr.remove_object(self.obj_info["name"])

    def detach_object_from_robot(self):
        """
        Detach object collision from robot collision
        """
        self.robot_col_mngr.remove_object(
            self.obj_info["name"]
        )

    def reattach_object(self, transform):
        """
        Reattach object collision
        """
        self.object_col_mngr.add_object(
            self.obj_info["name"], 
            gtype=self.obj_info["gtype"], 
            gparam=self.obj_info["gparam"], 
            h_mat=transform)

    @abstractclassmethod
    def get_joint_path(self):
        """
        write planner algorithm you want 
        """
        raise NotImplementedError

    @abstractclassmethod
    def _get_linear_path(self, init_pose, goal_pose):
        """
        Base Planner class 

        Args:
            init_pose (np.array): init robots' eef pose
            goal_pose (np.array): goal robots' eef pose        
        """
        raise NotImplementedError

    def _setup_collision_manager(
        self, 
        robot_col_manager,
        object_col_manager,
        is_attached,
        current_obj_info,
        result_obj_info,
        T_between_gripper_and_obj
    ):
        """
        Setup Robot, Object Collision Manager

        Args:
            robot_col_manager (CollisionManager): robot's CollisionManager
            object_col_manager (CollisionManager): object's CollisionManager
            is_attached (bool): if the object is attached or not
            current_obj_info (dict): current object info
            result_obj_info (dict): result object info
            T_between_gripper_and_obj (np.array): The transformation relationship between gripper and object.
        """

        self.robot_col_mngr = robot_col_manager
        self.object_col_mngr = object_col_manager
        self.is_attached = is_attached

        if current_obj_info is not None and result_obj_info is not None:
            self.obj_info = current_obj_info
            self.T_between_gripper_and_obj = T_between_gripper_and_obj
            self.result_object_pose = result_obj_info["transform"]

            if not self.is_attached:
                self.object_col_mngr.set_transform(self.obj_info["name"], self.obj_info["transform"])    

            self.robot_col_mngr.show_collision_info(name="Robot")
            self.object_col_mngr.show_collision_info(name="Object")

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
            self.q_limits_lower = self.robot.joint_limits_lower[self.arm]
            self.q_limits_upper = self.robot.joint_limits_upper[self.arm]
        else:
            self.q_limits_lower = self.robot.joint_limits_lower
            self.q_limits_upper = self.robot.joint_limits_upper

    def _check_q_in_limits(self, q_in):
        """
        check q_in within joint limits
        If q_in is in joint limits, return True
        otherwise, return False

        Returns:
            bool (True or False)
        """
        return np.all([q_in >= self.q_limits_lower, q_in <= self.q_limits_upper])

    def _check_robot_col_mngr(self, robot_col_manager):
        if robot_col_manager is None:
            return False
        
        is_self_collision = robot_col_manager.in_collision_internal()
        
        if is_self_collision:
            raise CollisionError("Conflict confirmed. Check the joint settings again")
        
        return True

    def _setup_eef_name(self):
        """
        Setup end-effector name
        """
        if self.arm is not None:
            self.eef_name = self.robot.eef_name[self.arm]

    def _collision_free(
        self, 
        new_q, 
        is_attached=False, 
        visible_name=False
    ):
        """
        Check collision free between robot and objects

        Args:
            new_q (np.array): new joint angles
            is_attached (bool): if the object is attached or not
            visible_name (bool): If it's true, the result of the collision and the name will come out. 
                                 Otherwise, only the collision results will come out.

        Returns:
            result (bool): If collision free, return True
            names (set of 2-tup): The set of pairwise collisions. 
        """
 
        if self.robot_col_mngr is None:
            return True

        fk = self._get_fk(new_q)
        
        if is_attached:
            grasp_pose = fk[self.robot.eef_name].h_mat
            obj_pose = np.dot(grasp_pose, self.T_between_gripper_and_obj)
            self.robot_col_mngr.set_transform(self.obj_info["name"], obj_pose)
        
        for link, transformation in fk.items():
            if link in self.robot_col_mngr._objs:
                transform = transformation.h_mat
                if self.robot_col_mngr.geom == "visual":
                    h_mat = np.dot(transform, self.robot.links[link].visual.offset.h_mat)
                else:
                    h_mat = np.dot(transform, self.robot.links[link].collision.offset.h_mat)
                self.robot_col_mngr.set_transform(name=link, h_mat=h_mat)
        
        is_self_collision = self.robot_col_mngr.in_collision_internal(return_names=False, return_data=False)
        
        if visible_name:
            is_object_collision, col_name = self.robot_col_mngr.in_collision_other(other_manager=self.object_col_mngr, return_names=visible_name)  
            if is_self_collision or is_object_collision:
                print(col_name)
                return False, col_name
            return True, col_name
            

        is_object_collision = self.robot_col_mngr.in_collision_other(other_manager=self.object_col_mngr, return_names=False)  
        if is_self_collision or is_object_collision:
            return False
        return True

    def _get_fk(self, q_in):
        """
        Get forward kinematics corresponding to q_in

        Args:
            q_in (np.array): joint angles

        Returns:
            fk (OrderedDict)
        """
        if self.robot.robot_name == "sawyer":
            q_in = np.concatenate((np.zeros(1), q_in))

        if self.arm is not None:
            fk = self.robot.forward_kin(q_in, self.robot.desired_frames[self.arm])
        else:
            fk = self.robot.forward_kin(q_in)
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