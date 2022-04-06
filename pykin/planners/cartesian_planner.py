import numpy as np


import pykin.utils.transform_utils as t_utils
import pykin.utils.kin_utils as k_utils
import pykin.kinematics.jacobian as jac

from pykin.planners.planner import Planner
from pykin.utils.kin_utils import ShellColors as sc, logging_time
from pykin.utils.log_utils import create_logger
from pykin.utils.transform_utils import get_linear_interpoation, get_quaternion_slerp

logger = create_logger('Cartesian Planner', "debug")

class CartesianPlanner(Planner):
    """
    path planner in Cartesian space

    Args:
        robot(SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        n_step(int): Number of waypoints
        dimension(int): robot arm's dof
        damping(float): Value using DLS(Damped Least Squares)
        epsilon(float): Threshold of pose error
        pos_sensitivity(float): Error reference value between the target pose and the current pose
        waypoint_type(str): Type of waypoint ex) "Linear", "Cubic", "Circular"
        is_slerp(bool): flag of quaternion slerp
    """
    def __init__(
        self,
        robot,
        n_step=500,
        dimension=7,
        damping=0.5,
        epsilon=1e-12,
        pos_sensitivity=0.03,
        waypoint_type="Linear",
        is_slerp=False
    ):
        super(CartesianPlanner, self).__init__(
            robot, 
            dimension)
            
        self.n_step = n_step
        self.waypoint_type = waypoint_type
        self.eef_name = self.robot.eef_name
        self._dimension = dimension
        self._damping = damping
        self._epsilon = epsilon
        self._pos_sensitivity = pos_sensitivity
        self._is_slerp = is_slerp

        self.arm = None

        super()._setup_q_limits()
        super()._setup_eef_name()
    
    def __repr__(self):
        return 'pykin.planners.cartesian_planner.{}()'.format(type(self).__name__)
    
    @logging_time
    def get_joint_path(
        self, 
        cur_q,
        goal_pose,
        resolution=1, 
        robot_col_manager=None,
        object_col_manager=None,
        is_attached=False, 
        current_obj_info=None,
        result_obj_info=None,
        T_between_gripper_and_obj=None,
    ):
        """
        Get joint paths in joint space

        Args:
            cur_q (sequence of float): current joints
            goal_pose (sequence of float): goal pose
            resolution (float): Get number of waypoints * resolution
            robot_col_manager (CollisionManager): robot's CollisionManager
            object_col_manager (CollisionManager): object's CollisionManager
            is_attached (bool): if the object is attached or not
            current_obj_info (dict): current object info
            result_obj_info (dict): result object info
            T_between_gripper_and_obj (np.array): The transformation relationship between gripper and object
        """
        logger.info(f"Start to compute Cartesian Planning")

        self._cur_qpos = super()._convert_numpy_type(cur_q)
        self._goal_pose = super()._convert_numpy_type(goal_pose)
        init_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, self._cur_qpos)
        self._cur_pose = self.robot.get_eef_pose(init_fk)
        self._resolution = resolution

        if not super()._check_robot_col_mngr(robot_col_manager):
            logger.warning(f"This Planner does not do collision checking")
            
        super()._setup_collision_manager(
            robot_col_manager,
            object_col_manager,
            is_attached,
            current_obj_info,
            result_obj_info,
            T_between_gripper_and_obj
        )
        
        waypoints = self.generate_waypoints()
        paths, target_positions = self._compute_paths_and_target_positions(waypoints)
        
        return paths, target_positions

    def _compute_paths_and_target_positions(self, waypoints):
        """
        Compute joint paths and target positions

        Args:
            waypoints (list): waypoints of eef's target pose 

        Returns:
            paths (list): list of joint position
            target_positions (list): list of eef's target position
        """
        cnt = 0
        total_cnt = 10
        while True:
            cnt += 1
            collision_pose = {}
            cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, self._cur_qpos)

            current_transform = cur_fk[self.eef_name].h_mat
            eef_position = cur_fk[self.eef_name].pos

            paths = [self._cur_qpos]
            target_positions = [eef_position]

            for step, (pos, ori) in enumerate(waypoints):
                target_transform = t_utils.get_h_mat(pos, ori)
                err_pose = k_utils.calc_pose_error(target_transform, current_transform, self._epsilon) 
                J = jac.calc_jacobian(self.robot.desired_frames, cur_fk, self._dimension)
                J_dls = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + self._damping**2 * np.identity(6)))

                dq = np.dot(J_dls, err_pose)
                self._cur_qpos = np.array([(self._cur_qpos[i] + dq[i]) for i in range(self._dimension)]).reshape(self._dimension,)
                if not self._check_q_in_limits(self._cur_qpos):
                    continue
                
                is_collision_free, col_name = self._collision_free(self._cur_qpos, self.is_attached, visible_name=True)
                if not is_collision_free:
                    collision_pose[step] = (col_name, np.round(target_transform[:3,3], 6))
                    continue

                cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, self._cur_qpos)
                current_transform = cur_fk[self.robot.eef_name].h_mat

                if step % (1/self._resolution) == 0 or step == len(waypoints)-1:
                    paths.append(self._cur_qpos)
                    target_positions.append(pos)

            err = t_utils.compute_pose_error(self._goal_pose[:3], cur_fk[self.eef_name].pos)
            
            if collision_pose.keys():
                logger.error(f"Failed Generate Path.. Collision may occur.")
                for col_name, _ in collision_pose.values():
                    logger.warning(f"\n\tCollision Names : {col_name}")
                paths, target_positions = [], []
                break

            if cnt > total_cnt:
                logger.error(f"Failed Generate Path.. The number of retries of {cnt} exceeded")
                paths, target_positions = [], []
                break
            
            if err < self._pos_sensitivity:
                logger.info(f"Generate Path Successfully!! Error is {err:6f}")
                break

            logger.error(f"Failed Generate Path.. Position Error is {err:6f}")
            print(f"{sc.BOLD}Retry Generate Path, the number of retries is {cnt}/{total_cnt} {sc.ENDC}\n")
        return paths, target_positions

    # TODO
    # generate cubic, circular waypoints
    def generate_waypoints(self):
        """
        Generate waypoints of eef's target pose

        Returns:
            waypoints (list): waypoints of eef's target pose 
        """
        if self.waypoint_type == "Linear":
            waypoints = [path for path in self._get_linear_path(self._cur_pose, self._goal_pose, self._is_slerp)]
        if self.waypoint_type == "Cubic":
            pass
        if self.waypoint_type == "Circular":
            pass
        return waypoints

    def _get_linear_path(self, init_pose, goal_pose, is_slerp):
        """
        Get linear path

        Args:
            init_pose (np.array): init robots' eef pose
            goal_pose (np.array): goal robots' eef pose 
            is_slerp (bool): flag of quaternion slerp      
        
        Return:
            pos, ori (tuple): position, orientation
        """
        for step in range(1, self.n_step + 1):
            delta_t = step / self.n_step
            pos = get_linear_interpoation(init_pose[:3], goal_pose[:3], delta_t)
            ori = init_pose[3:]
            if is_slerp:
                ori = get_quaternion_slerp(init_pose[3:], goal_pose[3:], delta_t)

            yield (pos, ori)

    def _get_cubic_path(self):
        pass

    def _get_cicular_path(self):
        pass

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution

    @property
    def damping(self):
        return self._damping
    
    @damping.setter
    def damping(self, damping):
        self._damping = damping

    @property
    def pos_sensitivity(self):
        return self._pos_sensitivity
    
    @pos_sensitivity.setter
    def pos_sensitivity(self, pos_sensitivity):
        self._pos_sensitivity = pos_sensitivity

    @property
    def is_slerp(self):
        return self._is_slerp
    
    @is_slerp.setter
    def is_slerp(self, is_slerp):
        self._is_slerp = is_slerp