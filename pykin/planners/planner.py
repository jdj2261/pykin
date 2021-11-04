import numpy as np
from abc import ABC, abstractclassmethod

from pykin.utils.fcl_utils import FclManager
from pykin.utils.kin_utils import get_robot_geom
from pykin.utils.error_utils import CollisionError, NotFoundError
from pykin.utils.transform_utils import get_homogeneous_matrix

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
        obstacles
    ):
        self.robot = robot
        self.obstacles = obstacles
        self.fcl_manager = FclManager()

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

    def _setup_fcl_manager(self, transformatios):
        """
        Setup fcl manager for collision checking
        """
        self._apply_fcl_to_robot(transformatios)
        self._apply_fcl_to_obstacles()

    def _apply_fcl_to_robot(self, transformatios):
        """
        Apply fcl to robot 
        """
        for link, transformation in transformatios.items():
            name, gtype, gparam = get_robot_geom(self.robot.links[link])
            transform = transformation.homogeneous_matrix
            self.fcl_manager.add_object(name, gtype, gparam, transform)
    
    def _apply_fcl_to_obstacles(self):
        """
        Apply fcl to obstacles 
        """
        if self.obstacles:
            for key, vals in self.obstacles:
                obs_type = vals[0]
                obs_param = vals[1]
                obs_pos = vals[2]
                ob_transform = get_homogeneous_matrix(position=np.array(obs_pos))
                self.fcl_manager.add_object(key, obs_type, obs_param, ob_transform)

    def _check_init_collision(self, goal_q=None):
        """
        Check collision between robot and obstacles
        """
        is_collision, obj_names = self.fcl_manager.collision_check(return_names=True)
        if is_collision:
            for name1, name2 in obj_names:
                if not ("obstacle" in name1 and "obstacle" in name2):
                    raise CollisionError(obj_names)

        if goal_q is not None:
            goal_collision_free, collision_names = self.collision_free(goal_q, visible_name=True)
            if not goal_collision_free:
                for name1, name2 in collision_names:
                    if ("obstacle" in name1 and "obstacle" not in name2) or \
                    ("obstacle" not in name1 and "obstacle" in name2):
                        raise CollisionError(collision_names)

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
            if link in self.fcl_manager._objs:
                transform = transformations.homogeneous_matrix
                self.fcl_manager.set_transform(name=link, transform=transform)

        is_collision, name = self.fcl_manager.collision_check(return_names=True, return_data=False)
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