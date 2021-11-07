import numpy as np
from abc import ABC, abstractclassmethod

from pykin.utils.collision_utils import get_robot_collision_geom, get_robot_visual_geom

from pykin.utils.log_utils import create_logger
from pykin.utils.error_utils import CollisionError, NotFoundError
from pykin.utils.transform_utils import get_h_mat, get_transform_to_visual

logger = create_logger('Cartesian Planner', "debug",)

class Planner(ABC):
    """
    Base Planner class 

    Args:
        robot (SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        obstacles (dictionary) : The obstacles
    """
    def __init__(
        self,
        robot,
        obstacles,
        collision_manager
    ):
        self.robot = robot
        self.obstacles = obstacles

        # TODO
        if collision_manager is None:
            logger.warning(f"This Planner does not do collision checking")
        else:
            self.collision_manager = collision_manager

            check_collision = self.collision_manager.collision_check()
            if check_collision:
                raise CollisionError("Conflict confirmed. Check the joint settings again")

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
        transformations = self._get_transformations(new_q)
        for link, transformations in transformations.items():
            if link in self.collision_manager._objs:
                transform = transformations.h_mat
                self.collision_manager.set_transform(name=link, transform=transform)
        is_collision = self.collision_manager.collision_check(return_names=False, return_data=False)

        name = None
        if visible_name:
            if is_collision:
                return False, name
            return True, name

        if is_collision:
            return False
        return True

    def _get_transformations(self, q_in):
        """
        Get transformations corresponding to q_in

        Args:
            q_in(np.array): joint angles

        Returns:
            transformations(OrderedDict)
        """
        if self.arm is not None:
            transformations = self.robot.forward_kin(q_in, self.robot.desired_frames[self.arm])
        else:
            transformations = self.robot.forward_kin(q_in, self.robot.desired_frames)
        return transformations

    @abstractclassmethod
    def get_path_in_joinst_space(self):
        """
        write planner algorithm you want 
        """
        raise NotImplementedError