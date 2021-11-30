import numpy as np
from abc import abstractclassmethod, ABCMeta

from pykin.utils.log_utils import create_logger
from pykin.utils.error_utils import CollisionError, NotFoundError

logger = create_logger('Cartesian Planner', "debug",)

class Planner(metaclass=ABCMeta):
    """
    Base Planner class 

    Args:
        robot (SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        obstacles (dictionary) : The obstacles
    """
    def __init__(
        self,
        robot,
        self_collision_manager,
        obstacle_collision_manager,
        dimension
    ):
        self.robot = robot
        self._dimension = dimension
        if self_collision_manager is None:
            logger.warning(f"This Planner does not do collision checking")
            self.self_c_manager = None
        else:
            self.self_c_manager = self_collision_manager
            check_collision = self.self_c_manager.in_collision_internal()
            if check_collision:
                raise CollisionError("Conflict confirmed. Check the joint settings again")
        self.obstacle_c_manager = obstacle_collision_manager

    def __repr__(self) -> str:
        return 'pykin.planners.planner.{}()'.format(type(self).__name__)

    @staticmethod
    def _change_types(datas):
        """
        """
        if not isinstance(datas, (np.ndarray)):
            datas = np.array(datas)
            if datas.size == 0:
                raise NotFoundError("Make sure set current or goal joints..")
        return datas

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
            bool(True or False)
        """
        return np.all([q_in >= self.q_limits_lower, q_in <= self.q_limits_upper])

    def _setup_eef_name(self):
        """
        Setup end-effector name
        """
        if self.arm is not None:
            self.eef_name = self.robot.eef_name[self.arm]

    @abstractclassmethod
    def collision_free(self, new_q, visible_name=False):
        """
        Check collision free between robot and obstacles
        If visible name is True, return collision result and collision object names
        otherwise, return only collision result

        Args:
            new_q(np.array): new joint angles
            visible_name(bool)

        Returns:
            result(bool): If collision free, return True
            names(set of 2-tup): The set of pairwise collisions. 
        """
        raise NotImplementedError

    @abstractclassmethod
    def get_path_in_joinst_space(self):
        """
        write planner algorithm you want 
        """
        raise NotImplementedError

    @abstractclassmethod
    def _get_linear_path(self, init_pose, goal_pose):
        raise NotImplementedError

    def _get_transformations(self, q_in):
        """
        Get transformations corresponding to q_in

        Args:
            q_in(np.array): joint angles

        Returns:
            transformations(OrderedDict)
        """
        if self.robot.robot_name == "sawyer":
            q_in = np.concatenate((np.zeros(1), q_in))

        if self.arm is not None:
            transformations = self.robot.forward_kin(q_in, self.robot.desired_frames[self.arm])
        else:
            transformations = self.robot.forward_kin(q_in)
        return transformations

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