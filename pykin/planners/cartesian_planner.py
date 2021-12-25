import numpy as np


import pykin.utils.transform_utils as t_utils
import pykin.utils.kin_utils as k_utils
import pykin.kinematics.jacobian as jac

from pykin.planners.planner import Planner
from pykin.utils.error_utils import OriValueError, CollisionError
from pykin.utils.kin_utils import ShellColors as sc
from pykin.utils.log_utils import create_logger
from pykin.utils.transform_utils import get_linear_interpoation, get_quaternion_slerp

logger = create_logger('Cartesian Planner', "debug",)

class CartesianPlanner(Planner):
    """
    path planner in Cartesian space

    Args:
        robot(SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
    """
    def __init__(
        self,
        robot,
        self_collision_manager=None,
        obstacle_collision_manager=None,
        n_step=500,
        dimension=7,
        waypoint_type="Linear"
    ):
        super(CartesianPlanner, self).__init__(
            robot, 
            self_collision_manager, 
            obstacle_collision_manager,
            dimension)
        self.n_step = n_step
        self.waypoint_type = waypoint_type
        self.eef_name = self.robot.eef_name
        self.arm = None
        self._dimension = dimension

        super()._setup_q_limits()
        super()._setup_eef_name()
    
    def __repr__(self):
        return 'pykin.planners.cartesian_planner.{}()'.format(type(self).__name__)
        
    def get_path_in_joinst_space(
        self, 
        current_q=None,
        goal_pose=None,
        waypoints=None,
        resolution=1, 
        damping=0.5,
        epsilon=1e-12,
        pos_sensitivity=0.03,
        is_slerp=False
    ):
        self._cur_qpos = super()._change_types(current_q)
        self._goal_pose = super()._change_types(goal_pose)
        init_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, self._cur_qpos)
        self._cur_pose = self.robot.get_eef_pose(init_fk)

        self._resolution = resolution
        self._damping = damping
        self._pos_sensitivity = pos_sensitivity
        self._is_slerp = is_slerp

        if waypoints is None:
            waypoints = self.generate_waypoints(is_slerp)

        paths, target_positions = self._compute_path_and_target_pose(waypoints, epsilon)
        
        return paths, target_positions

    def _compute_path_and_target_pose(self, waypoints, epsilon):
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
                err_pose = k_utils.calc_pose_error(target_transform, current_transform, epsilon) 
                J = jac.calc_jacobian(self.robot.desired_frames, cur_fk, self._dimension)
                J_dls = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + self._damping**2 * np.identity(6)))

                dq = np.dot(J_dls, err_pose)
                self._cur_qpos = np.array([(self._cur_qpos[i] + dq[i]) for i in range(self._dimension)]).reshape(self._dimension,)

                collision_free = self.collision_free(self._cur_qpos, visible_name=False)

                if not collision_free:
                    collision_pose[step] = np.round(target_transform[:3,3],3)
                    continue

                if not self._check_q_in_limits(self._cur_qpos):
                    continue

                cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, self._cur_qpos)
                current_transform = cur_fk[self.robot.eef_name].h_mat

                if step % (1/self._resolution) == 0 or step == len(waypoints)-1:
                    paths.append(self._cur_qpos)
                    target_positions.append(pos)

            err = t_utils.compute_pose_error(self._goal_pose[:3], cur_fk[self.eef_name].pos)
            
            if collision_pose.keys():
                logger.error(f"Failed Generate Path.. Collision may occur.")
                # logger.warning(f"Collision Position : {collision_pose.values()}")
                for pose in collision_pose.values():
                    logger.warning(f"Collision Position : {pose}")
                raise CollisionError("Conflict confirmed. Check the object position!")
                
            if err < self._pos_sensitivity:
                logger.info(f"Generate Path Successfully!! Error is {err:6f}")
                break

            if cnt > total_cnt:
                logger.error(f"Failed Generate Path.. The number of retries of {cnt} exceeded")
                paths, target_positions = None, None
                break

            logger.error(f"Failed Generate Path.. Position Error is {err:6f}")
            print(f"{sc.BOLD}Retry Generate Path, the number of retries is {cnt}/{total_cnt} {sc.ENDC}\n")
            
        return paths, target_positions

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

        if self.self_c_manager is None:
            return True
            
        transformations = self._get_transformations(new_q)
        for link, transformations in transformations.items():
           if link in self.self_c_manager._objs:
                transform = transformations.h_mat
                A2B = np.dot(transform, self.robot.links[link].visual.offset.h_mat)
                self.self_c_manager.set_transform(name=link, transform=A2B)
        is_self_collision = self.self_c_manager.in_collision_internal(return_names=False, return_data=False)
        is_obstacle_collision = self.self_c_manager.in_collision_other(other_manager=self.obstacle_c_manager, return_names=False)

        name = None
        if visible_name:
            if is_self_collision:
                return False, name
            return True, name

        if is_self_collision or is_obstacle_collision:
            return False
        return True

    # TODO
    # generate cubic, circular waypoints
    def generate_waypoints(self, is_slerp):
        if self.waypoint_type == "Linear":
            waypoints = [path for path in self._get_linear_path(self._cur_pose, self._goal_pose, is_slerp)]
        if self.waypoint_type == "Cubic":
            pass
        if self.waypoint_type == "Circular":
            pass
        return waypoints

    def get_waypoints(self):
        return self.waypoints

    def _change_pose_type(self, pose):
        ret = np.zeros(7)
        ret[:3] = pose[:3]
        
        if isinstance(pose, (list, tuple)):
            pose = np.asarray(pose)
        ori = pose[3:]

        if ori.shape == (3,):
            ori = t_utils.get_quaternion_from_rpy(ori)
            ret[3:] = ori
        elif ori.shape == (4,):
            ret[3:] = ori
        else:
            raise OriValueError(ori.shape)

        return ret

    def _get_linear_path(self, init_pose, goal_pose, is_slerp):
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