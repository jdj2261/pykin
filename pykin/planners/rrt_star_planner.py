import math
import numpy as np

from pykin.planners.planner import Planner
from pykin.planners.tree import Tree
from pykin.utils.fcl_utils import FclManager
from pykin.utils.kin_utils import get_robot_geom
from pykin.utils.error_utils import NotFoundError, CollisionError
from pykin.utils.transform_utils import get_homogeneous_matrix

class RRTStarPlanner(Planner):
    """
    RRT star spath planning

    Args:
        robot(SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        obstacles(Obstacle): The obstacles
        current_q(np.array or Iterable of floats): current joint angle
        goal_q(np.array or Iterable of floats): target joint angle obtained through Inverse Kinematics 
        delta_distance(float): distance between nearest vertex and new vertex
        epsilon(float): 1-epsilon is probability of random sampling
        max_iter(int): maximum number of iterations
        gamma_RRT_star(int): factor used for search radius
    """
    def __init__(
        self, 
        robot,
        obstacles,
        current_q=None,
        goal_q=None,
        delta_distance=0.5,
        epsilon=0.2,
        max_iter=3000,
        gamma_RRT_star=300, # At least gamma_RRT > delta_distance,
    ):
        super(RRTStarPlanner, self).__init__(robot, obstacles)
      
        self.cur_q = current_q
        self.goal_q  = goal_q
        self.delta_dis = delta_distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gamma_RRTs = gamma_RRT_star
        
        self.T = None
        self.cost = None

        self.arm = None
        self.dimension = self.robot.dof
        self.eef_name = self.robot.eef_name

    def setup_start_goal_joint(
        self, 
        current_q, 
        goal_q, 
        arm=None, 
        init_transformation=None
    ):
        """
        Setup start joints and goal joints

        Args:
            current_q(np.array or Iterable of floats): current joint angle
            goal_q(np.array or Iterable of floats): target joint angle obtained through Inverse Kinematics 
            arm(str): If Manipulator types is bimanul, must put the arm type.
            init_transformation: Initial transformations
        """
        if self._check_q_datas(current_q, goal_q):
            self.cur_q = current_q
            self.goal_q  = goal_q

        self.arm = arm

        if init_transformation is None:
            init_transformation = self.robot.init_transformations

        self.dimension = len(current_q)

        self._setup_q_limits()
        self._setup_eef_name()
        self._setup_fcl_manager(init_transformation)
        self._check_init_collision()

    @staticmethod
    def _check_q_datas(current_q, goal_q):
        """
        Check input current joints and goal joints

        Args:
            current_q(np.array or Iterable of floats): current joint angle
            goal_q(np.array or Iterable of floats): target joint angle obtained through Inverse Kinematics
            
        Return:
            True(bool): Return True if everything is confirmed
        """
        if not isinstance(current_q, (np.ndarray)):
            current_q = np.array(current_q)
        
        if not isinstance(goal_q, (np.ndarray)):
            goal_q = np.array(goal_q)
 
        if current_q.size == 0 or goal_q.size == 0:
            raise NotFoundError("Make sure set current or goal joints..")
        return True

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
        self.fcl_manager = FclManager()
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

    def _check_init_collision(self):
        """
        Check collision between robot and obstacles
        """
        is_collision, obj_names = self.fcl_manager.collision_check(return_names=True)
        if is_collision:
            for name1, name2 in obj_names:
                if not ("obstacle" in name1 and "obstacle" in name2):
                    raise CollisionError(obj_names)

        goal_collision_free, collision_names = self.collision_free(self.goal_q, visible_name=True)
        if not goal_collision_free:
            for name1, name2 in collision_names:
                if ("obstacle" in name1 and "obstacle" not in name2) or \
                   ("obstacle" not in name1 and "obstacle" in name2):
                   raise CollisionError(collision_names)

    def generate_path(self):
        """
        Generate planner path 

        Returns:
            path(list) : result path (from start joints to goal joints)
        """
        path = None
        self.T = Tree()
        self.cost = {}

        self.T.add_vertex(self.cur_q)
        self.cost[0] = 0

        for k in range(self.max_iter):
            if k % 300 == 0 and k !=0:
                print(f"iter : {k}")
                
            rand_q = self.random_state()
            if not self.collision_free(rand_q):
                continue

            nearest_q, nearest_idx = self.nearest_neighbor(rand_q, self.T)
            new_q = self.new_state(nearest_q, rand_q)
   
            if self.collision_free(new_q) and self._check_q_in_limits(new_q):
                neighbor_indexes = self.get_near_neighbor_indices(new_q)
                min_cost = self.get_new_cost(nearest_idx, nearest_q, new_q)
                min_cost, nearest_idx = self.get_minimum_cost_and_index(neighbor_indexes, new_q, min_cost, nearest_idx)
 
                self.T.add_vertex(new_q)
                new_idx = len(self.T.vertices) - 1
                self.cost[new_idx] = min_cost
                self.T.add_edge([nearest_idx, new_idx])

                self.rewire(neighbor_indexes, new_q, new_idx)

                if self.reach_to_goal(new_q):                    
                    path = self.find_path(self.T)
        return path

    def random_state(self):
        """
        sampling joints in q space within joint limits 
        If random probability is greater than the epsilon, return random joint angles
        oterwise, return goal joint angles

        Returns:
            q_outs(np.array) : 
        """
        q_outs = np.zeros(self.dimension)
        
        if np.random.random() > self.epsilon:
            for i, (q_min, q_max) in enumerate(zip(self.q_limits_lower, self.q_limits_upper)):
                q_outs[i] = np.random.uniform(q_min, q_max)
        else:
            q_outs = self.goal_q

        return q_outs

    def nearest_neighbor(self, random_q, tree):
        """
        Find nearest neighbor vertex and index from random_q

        Args:
            random_q(np.array): sampled random joint angles 
            tree(Tree): Trees obtained so far

        Returns:
            nearest_vertex(np.array): nearest vertex(joint angles)
            nearest_idx(int): nearest index
        """
        distances = [self.distance(random_q, vertex) for vertex in tree.vertices]
        nearest_idx = np.argmin(distances)
        nearest_vertex = tree.vertices[nearest_idx]
        return nearest_vertex, nearest_idx

    def distance(self, pointA, pointB):
        """
        Get distance from pointA to pointB

        Args:
            pointA(np.array)
            pointB(np.array)
            
        Returns:
            Norm(float or ndarray)
        """
        return np.linalg.norm(pointB - pointA)

    def new_state(self, nearest_q, random_q):
        """
        Get new point between nearest vertex and random vertex

        Args:
            nearest_q(np.array): nearest joint angles 
            random_q(np.array): sampled random joint angles 

        Returns:
            new_q(np.array): new joint angles
        """
        if np.equal(nearest_q, random_q).all():
            return nearest_q

        vector = random_q - nearest_q
        dist = self.distance(random_q, nearest_q)
        step = min(self.delta_dis, dist)
        unit_vector = vector / dist
        new_q = nearest_q + unit_vector * step

        return new_q

    def get_near_neighbor_indices(self, q):
        """
        Returns all neighbor indices within the search radius from the new vertex

        Args:
            q(np.array): new joint angles 

        Returns:
            near_indexes(list): all neighbor indices
        """
        card_V = len(self.T.vertices) + 1
        r = self.gamma_RRTs * ((math.log(card_V) / card_V) ** (1/self.dimension))

        search_radius = min(r, self.delta_dis)
        dist_list = [self.distance(vertex, q) for vertex in self.T.vertices]
                                                   
        near_indexes = []
        for idx, dist in enumerate(dist_list):
            if dist <= search_radius:
                near_indexes.append(idx)

        return near_indexes

    def get_new_cost(self, idx, pointA, pointB):
        """
        Returns new cost 

        Args:
            idx(int): neighbor vertex's index
            A(np.array)
            B(np.array)

        Returns:
            cost(float)
        """
        cost = self.cost[idx] + self.distance(pointA, pointB)
        return cost

    def get_minimum_cost_and_index(self, neighbor_indexes, new_q, min_cost, nearest_idx):
        """
        Returns minimum cost and neer vertex index 
        between neighbor vertices and new vertex in search radius

        Args:
            neighbor_indexes: neighbor vertex's index
            new_q(int): new joint angles
            min_cost(np.array): minimum cost
            nearest_idx(np.array): nearest index

        Returns:
            min_cost(float)
            nearest_idx(int)
        """
        for i in neighbor_indexes:
            new_cost = self.get_new_cost(i, new_q, self.T.vertices[i])

            if new_cost < min_cost:
                min_cost = new_cost
                nearest_idx = i

        return min_cost, nearest_idx

    def rewire(self, neighbor_indexes, new_q, new_idx):
        """
        Rewire a new vertex with a neighbor vertex with minimum cost

        Args:
            neighbor_indexes: neighbor vertex's index
            new_q(int): new joint angles
            new_idx(np.array): new joint angles's index
        """
        for i in neighbor_indexes:
            new_cost = self.get_new_cost(new_idx, new_q, self.T.vertices[i])

            if new_cost < self.cost[i]:
                self.cost[i] = new_cost
                self.T.edges[i-1][0] = new_idx

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

    def reach_to_goal(self, point):
        """
        Check reach to goal
        If reach to goal, return True
        Args:
            point(np.array): joint angles

        Returns:
            bool
        """
        dist = self.distance(point, self.goal_q)
        if dist <= 0.5:
            return True
        return False

    def find_path(self, tree):
        """
        find path result from start index to goal index

        Args:
            tree(Tree): Trees obtained so far

        Returns:
            path(list)
        """
        path = [self.goal_q]
        goal_idx = tree.edges[-1][1]
 
        while goal_idx != 0:
            if not np.allclose(path[0], tree.vertices[goal_idx]):
                path.append(tree.vertices[goal_idx])
            parent_idx = tree.edges[goal_idx-1][0]
            goal_idx = parent_idx
        path.append(self.cur_q)

        return path[::-1]

    def get_rrt_tree(self):
        """
        Return obtained RRT Trees

        Returns:
            verteices(list)
        """
        vertices = []
        for edge in self.T.edges:
            from_node = self.T.vertices[edge[0]]
            goal_node = self.T.vertices[edge[1]]
            vertices.append((from_node, goal_node))
        return vertices
