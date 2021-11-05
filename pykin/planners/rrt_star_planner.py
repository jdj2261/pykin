import math
import numpy as np

from pykin.planners.joint_planner import JointPlanner
from pykin.planners.tree import Tree

class RRTStarPlanner(JointPlanner):
    """
    RRT star path planner

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

    def __repr__(self):
        return 'pykin.planners.rrt_star_planner.{}()'.format(type(self).__name__)
        
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
        self.cur_q = super()._change_types(current_q)
        self.goal_q = super()._change_types(goal_q)

        if init_transformation is None:
            init_transformation = self.robot.init_transformations

        self.arm = arm
        self.dimension = len(current_q)

        super()._setup_q_limits()
        super()._setup_eef_name()
        super()._setup_collision_manager(init_transformation)
        super()._check_init_collision(self.goal_q)

    def get_path_in_joinst_space(self):
        """
        Get path in joint space

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
