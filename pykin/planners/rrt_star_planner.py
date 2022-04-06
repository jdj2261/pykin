import math
import numpy as np
import networkx as nx

from scipy.spatial import distance


from pykin.planners.planner import NodeData, Planner
# from pykin.planners.tree import Tree
from pykin.utils.log_utils import create_logger
from pykin.utils.kin_utils import ShellColors as sc, logging_time
from pykin.utils.transform_utils import get_linear_interpoation

logger = create_logger('RRT Star Planner', "debug")

class RRTStarPlanner(Planner):
    """
    RRT star path planner

    Args:
        robot(SingleArm or Bimanual): manipulator type
        delta_distance(float): distance between nearest point and new point
        epsilon(float): 1-epsilon is probability of random sampling
        gamma_RRT_star(int): factor used for search radius
        max_iter(int): maximum number of iterations
        dimension(int): robot arm's dof
    """

    def __init__(
        self, 
        robot,
        delta_distance=0.5,
        epsilon=0.2,
        gamma_RRT_star=300, # At least gamma_RRT > delta_distance,
        dimension=7,
    ):
        super(RRTStarPlanner, self).__init__(
            robot, 
            dimension
        )
        self.delta_dis = delta_distance
        self.epsilon = epsilon
        self.gamma_RRTs = gamma_RRT_star
        
        self._max_iter = None
        self._cur_qpos = None
        self._goal_pose = None
        self.tree = None

        self.arm = None
        self.dimension = dimension
        self.eef_name = self.robot.eef_name

        super()._setup_q_limits()
        super()._setup_eef_name()

    def __repr__(self):
        return 'pykin.planners.rrt_star_planner.{}()'.format(type(self).__name__)

    def _create_tree(self):
        tree = nx.DiGraph()
        tree.add_node(0)
        tree.update(
            nodes=[(0, {NodeData.COST: 0,
                        NodeData.POINT: None})])
        return tree

    @logging_time
    def run(
        self, 
        cur_q,
        goal_pose, 
        max_iter=1000, 
        robot_col_manager=None,
        object_col_manager=None,
        is_attached=False, 
        current_obj_info=None,
        result_obj_info=None,
        T_between_gripper_and_obj=None,
    ):
        """
        Get path in joint space

        Args:
            cur_q (sequence of float): current joints
            goal_pose (sequence of float): goal pose
            max_iter(int): maximum number of iterations
            robot_col_manager (CollisionManager): robot's CollisionManager
            object_col_manager (CollisionManager): object's CollisionManager
            is_attached (bool): if the object is attached or not
            current_obj_info (dict): current object info
            result_obj_info (dict): result object info
            T_between_gripper_and_obj (np.array): The transformation relationship between gripper and object
        """
        logger.info(f"Start to compute RRT-star Planning")

        self._cur_qpos = super()._convert_numpy_type(cur_q)
        self._goal_pose = super()._convert_numpy_type(goal_pose)
        
        self._max_iter = max_iter

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

        cnt = 0
        total_cnt = 10

        while True:
            cnt += 1
            for _ in range(total_cnt):
                self.goal_q = self.robot.inverse_kin(np.random.randn(self._dimension), self._goal_pose)
                if self._check_q_in_limits(self.goal_q):
                    break
                if cnt > total_cnt:
                    logger.error(f"Failed Generate Path.. The number of retries of {cnt} exceeded")
                    break
                print(f"{sc.WARNING}Retry compute IK{sc.ENDC}")

            self.goal_node = None
            self.tree = self._create_tree()
            self.tree.nodes[0][NodeData.POINT] = self._cur_qpos

            for step in range(self._max_iter):
                if step % 100 == 0 and step !=0:
                    logger.info(f"iter : {step}")
                    
                q_rand = self._sample_free()
                if not self._collision_free(q_rand, is_attached):
                    continue
                
                nearest_node, q_nearest = self._nearest(q_rand)
                q_new = self._steer(q_nearest, q_rand)
    
                if self._collision_free(q_new, is_attached) and self._check_q_in_limits(q_new):
                    near_nodes = self._near(q_new)

                    new_node = self.tree.number_of_nodes()
                    self.tree.add_node(new_node)
                    
                    c_min = self.tree.nodes[nearest_node][NodeData.COST] + self._get_distance(q_nearest, q_new)
                    min_node = nearest_node

                    for near_node in near_nodes:
                        q_near = self.tree.nodes[near_node][NodeData.POINT]
                        near_cost = self.tree.nodes[near_node][NodeData.COST]
                        if (near_cost + self._get_distance(q_near, q_new)) < c_min:
                            c_min = near_cost + self._get_distance(q_near, q_new)
                            min_node = near_node

                    self.tree.update(nodes=[(new_node, {NodeData.COST: c_min,
                                                     NodeData.POINT: q_new})])
                    self.tree.add_edge(min_node, new_node)

                    new_cost = self.tree.nodes[new_node][NodeData.COST]
                    q_new = self.tree.nodes[new_node][NodeData.POINT]

                    # rewire
                    for near_node in near_nodes:
                        q_near = self.tree.nodes[near_node][NodeData.POINT]
                        near_cost = self.tree.nodes[near_node][NodeData.COST]
                        
                        if (new_cost + self._get_distance(q_near, q_new)) < near_cost:
                            parent_node = [node for node in self.tree.predecessors(near_node)][0]
                            self.tree.remove_edge(parent_node, near_node)
                            self.tree.add_edge(new_node, near_node)
                            print("rewire")
                    
                    if self._reach_to_goal(q_new):
                        self.goal_node = new_node

            if self.goal_node:
                logger.info(f"Generate Path Successfully!!")  
                break 

            if cnt > total_cnt:
                logger.error(f"Failed Generate Path.. The number of retries of {cnt} exceeded")
                break

            logger.error(f"Failed Generate Path..")
            print(f"{sc.BOLD}Retry Generate Path, the number of retries is {cnt}/{total_cnt} {sc.ENDC}\n")

    def get_joint_path(self, goal_node=None, n_step=1):
        """
        Get path in joint space

        Args:
            goal_node(int): goal node in rrt path
            n_step(int): number for n equal divisions between waypoints
    
        Returns:
            interpolate_paths(list) : interpoated paths from start joint pose to goal joint
        """
        
        path = [self.goal_q]
        if goal_node is None:
            goal_node = self.goal_node

        parent_node = [node for node in self.tree.predecessors(goal_node)][0]
        while parent_node:
            path.append(self.tree.nodes[parent_node][NodeData.POINT])
            parent_node = [node for node in self.tree.predecessors(parent_node)][0]
        
        path.append(self._cur_qpos)
        path.reverse()

        unique_path = []
        for joints in path:
            if not any(np.array_equal(np.round(joints, 8), np.round(unique_joints,8)) for unique_joints in unique_path):
                unique_path.append(joints)

        if n_step == 1:
            logger.info(f"Path Length : {len(unique_path)}")
            return unique_path

        interpolate_path = []
        interpolate_paths = []
        for i in range(len(unique_path)-1):
            interpolate_path = [unique_path.tolist() for unique_path in self._get_linear_path(unique_path[i], unique_path[i+1], n_step)]
            interpolate_paths.extend(interpolate_path)
        logger.info(f"Path length {len(unique_path)} --> {len(interpolate_paths)}")
        return interpolate_paths

    def get_rrt_tree(self):
        """
        Return obtained RRT Trees

        Returns:
            tree(list)
        """
        tree = []
        for edge in self.tree.edges:
            from_node = self.tree.vertices[edge[0]]
            goal_node = self.tree.vertices[edge[1]]
            tree.append((from_node, goal_node))
        return tree

    def _sample_free(self):
        """
        sampling joints in q space within joint limits 
        If random probability is greater than the epsilon, return random joint angles
        oterwise, return goal joint angles

        Returns:
            q_outs(np.array)
        """
        q_outs = np.zeros(self.dimension)
        
        if np.random.random() > self.epsilon:
            for i, (q_min, q_max) in enumerate(zip(self.q_limits_lower, self.q_limits_upper)):
                q_outs[i] = np.random.uniform(q_min, q_max)
        else:
            q_outs = self.goal_q

        return q_outs

    def _nearest(self, q_rand):
        """
        Find nearest neighbor point and index from q_rand

        Args:
            q_rand(np.array): sampled random joint angles 

        Returns:
            nearest_node(int): nearest node
            nearest_point(np.array): nearest point(joint angles)
        """
        distances = [self._get_distance(self.tree.nodes[node][NodeData.POINT], q_rand) for node in self.tree.nodes]
        nearest_node = np.argmin(distances)
        nearest_point = self.tree.nodes[nearest_node][NodeData.POINT]
        return nearest_node, nearest_point

    def _get_distance(self, p1, p2):
        """
        Get distance from pointA to pointB

        Args:
            p1(np.array)
            p2(np.array)
            
        Returns:
            Norm(float or ndarray)
        """

        return np.linalg.norm(p2-p1)
    
    def _steer(self, q_nearest, q_random):
        """
        Get new point between nearest point and random point

        Args:
            q_nearest(np.array): nearest joint angles 
            q_random(np.array): sampled random joint angles 

        Returns:
            q_new(np.array): new joint angles
        """
        if np.equal(q_nearest, q_random).all():
            return q_nearest

        vector = q_random - q_nearest
        dist = self._get_distance(q_random, q_nearest)
        step = min(self.delta_dis, dist)
        unit_vector = vector / np.linalg.norm(vector)
        q_new = q_nearest + unit_vector * step

        return q_new

    def _near(self, q_rand):
        """
        Returns all neighbor nodes within the search radius from the new point

        Args:
            q_rand(np.array): new joint angles 

        Returns:
            near_nodes(list): all neighbor nodes
        """
        card_V = len(self.tree.nodes) + 1
        r = self.gamma_RRTs * ((math.log(card_V) / card_V) ** (1/self._dimension))
        search_radius = min(r, self.gamma_RRTs)
        distances = [self._get_distance(self.tree.nodes[node][NodeData.POINT], q_rand) for node in self.tree.nodes]
                              
        near_nodes = []
        for node, dist in enumerate(distances):
            if dist <= search_radius:
                near_nodes.append(node)

        return near_nodes

    def _reach_to_goal(self, point):
        """
        Check reach to goal
        If reach to goal, return True
        Args:
            point(np.array): joint angles

        Returns:
            bool
        """
        dist = self._get_distance(point, self.goal_q)
        if dist <= 0.5:
            return True
        return False

    def _get_linear_path(self, init_pose, goal_pose, n_step=1):
        """
        Get linear path (only qpos)

        Args:
            init_pose (np.array): init robots' eef pose
            goal_pose (np.array): goal robots' eef pose  
            n_step(int): number for n equal divisions between waypoints
        
        Return:
            pos (np.array): position
        """
        for step in range(1, n_step + 1):
            delta_t = step / n_step
            pos = get_linear_interpoation(init_pose, goal_pose, delta_t)
            yield pos

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter