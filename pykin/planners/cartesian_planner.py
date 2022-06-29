import numpy as np
import random
import pykin.utils.transform_utils as t_utils
import pykin.utils.kin_utils as k_utils
import pykin.kinematics.jacobian as jac

from pykin.planners.planner import Planner
from pykin.utils.kin_utils import ShellColors as sc, logging_time
from pykin.utils.log_utils import create_logger
from pykin.utils.transform_utils import get_linear_interpoation, get_quaternion_slerp, get_quaternion_from_matrix

logger = create_logger('Cartesian Planner', "debug")

class CartesianPlanner(Planner):
    """
    path planner in Cartesian space

    Args:
        scene_mngr(SceneManager): Scene Manager
        n_step(int): Number of waypoints
        dimension(int): robot arm's dof
        damping(float): Value using DLS(Damped Least Squares)
        threshold(float): Threshold of pose error
        goal_tolerance(float): The tolerance in meters for the controller in the x & y distance when achieving a goal
        waypoint_type(str): Type of waypoint ex) "Linear", "Cubic", "Circular"
        is_slerp(bool): flag of quaternion slerp
    """
    def __init__(
        self,
        n_step=1000,
        dimension=7,
        damping=0.01,
        threshold=1e-12,
        goal_tolerance=0.1,
        waypoint_type="Linear",
        is_slerp=False
    ):
        super(CartesianPlanner, self).__init__(dimension)
            
        self._n_step = n_step
        self.waypoint_type = waypoint_type
        self._dimension = dimension
        self._damping = damping
        self._threshold = threshold
        self._goal_tolerance = goal_tolerance
        self._is_slerp = is_slerp

    def __repr__(self):
        return 'pykin.planners.cartesian_planner.{}()'.format(type(self).__name__)
    
    @logging_time
    def run(
        self,
        scene_mngr,
        cur_q,
        goal_pose,
        resolution=1,
        collision_check=True
    ):
        """
        Compute cartesian path

        Args:
            scene_mngr (SceneManager): pykin.scene.scene.SceneManager
            cur_q (sequence of float): current joints
            goal_pose (sequence of float): goal pose
            resolution (float): Get number of waypoints * resolution
        """
        if not scene_mngr:
            raise ValueError("SceneManager needs to be added first")
        logger.info(f"Start to compute Cartesian Planning")
        
        self._resolution = resolution
        self._scene_mngr = scene_mngr
        super()._setup_q_limits()
        super()._setup_eef_name()

        self._cur_qpos = super()._convert_numpy_type(cur_q)
        self._goal_pose = super()._convert_numpy_type(goal_pose)
        
        init_fk = self._scene_mngr.scene.robot.kin.forward_kinematics(self._scene_mngr.scene.robot.desired_frames, self._cur_qpos)
        self._cur_pose = self._scene_mngr.scene.robot.compute_eef_pose(init_fk)
        
        if not super()._check_robot_col_mngr():
            logger.warning(f"This Planner does not do collision checking")

        waypoints = self.generate_waypoints()
        joint_path = self._compute_paths_and_target_positions(waypoints, collision_check)

        self.joint_path = joint_path

    def get_joint_path(self):
        return self.joint_path

    def _compute_paths_and_target_positions(self, waypoints, collision_check=True):
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
        init_cur_qpos = self._cur_qpos
        while True:
            cnt += 1
            collision_pose = {}
            success_limit_check = True
            cur_fk = self._scene_mngr.scene.robot.kin.forward_kinematics(self._scene_mngr.scene.robot.desired_frames, init_cur_qpos)

            current_transform = cur_fk[self._scene_mngr.scene.robot.eef_name].h_mat
            joint_path = [init_cur_qpos]
            cur_qpos = init_cur_qpos

            for step, (pos, ori) in enumerate(waypoints):
                target_transform = t_utils.get_h_mat(pos, ori)
                err_pose = k_utils.calc_pose_error(target_transform, current_transform, self._threshold) 
                J = jac.calc_jacobian(self._scene_mngr.scene.robot.desired_frames, cur_fk, self._dimension)
                J_dls = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + self._damping**2 * np.identity(6)))

                dq = np.dot(J_dls, err_pose)
                cur_qpos = np.array([(cur_qpos[i] + dq[i]) for i in range(self._dimension)]).reshape(self._dimension,)
                
                if not self._check_q_in_limits(cur_qpos):
                    success_limit_check = False
                    
                if collision_check:
                    is_collide, col_name = self._collide(cur_qpos, visible_name=True)
                    if is_collide:
                        collision_pose[step] = (col_name, np.round(target_transform[:3,3], 6))
                        continue

                cur_fk = self._scene_mngr.scene.robot.kin.forward_kinematics(self._scene_mngr.scene.robot.desired_frames, cur_qpos)
                current_transform = cur_fk[self._scene_mngr.scene.robot.eef_name].h_mat

                if success_limit_check:
                    if step % (1/self._resolution) == 0 or step == len(waypoints)-1:
                        joint_path.append(cur_qpos)
                success_limit_check = True

            err = t_utils.compute_pose_error(self._goal_pose[:3,3], cur_fk[self._scene_mngr.scene.robot.eef_name].pos)
            
            if collision_pose.keys() and collision_check:
                logger.error(f"Failed Generate Path.. Collision may occur.")
                for col_name, _ in collision_pose.values():
                    logger.warning(f"\n\tCollision Names : {col_name}")
                joint_path = []
                break

            if cnt > total_cnt:
                logger.error(f"Failed Generate Path.. The number of retries of {cnt} exceeded")
                joint_path = []

                # ![DEBUG]
                self._scene_mngr.render_debug(title="Failed Cartesian Path")
                break
            
            if err < self._goal_tolerance:
                logger.info(f"Generate Path Successfully!! Error is {err:6f}")
                break

            logger.error(f"Failed Generate Path.. Position Error is {err:6f}")
            print(f"{sc.BOLD}Retry Generate Path, the number of retries is {cnt}/{total_cnt} {sc.ENDC}\n")
            self._damping = random.uniform(0, 0.1)
        
        return joint_path

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
        for step in range(1, self._n_step + 1):
            delta_t = step / self._n_step
            pos = get_linear_interpoation(init_pose[:3], goal_pose[:3, 3], delta_t)
            ori = init_pose[3:]
            if is_slerp:
                goal_q = get_quaternion_from_matrix(goal_pose[:3, :3])
                ori = get_quaternion_slerp(init_pose[3:], goal_q, delta_t)
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
    def goal_tolerance(self):
        return self._goal_tolerance
    
    @goal_tolerance.setter
    def goal_tolerance(self, goal_tolerance):
        self._goal_tolerance = goal_tolerance

    @property
    def is_slerp(self):
        return self._is_slerp
    
    @is_slerp.setter
    def is_slerp(self, is_slerp):
        self._is_slerp = is_slerp