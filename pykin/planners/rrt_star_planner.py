import math
import numpy as np

from pykin.planners.planner import Planner
from pykin.planners.tree import Tree
from pykin.utils.log_utils import create_logger
from pykin.utils.kin_utils import ShellColors as sc
from pykin.utils.transform_utils import get_linear_interpoation

logger = create_logger('RRT Star Planner', "debug")

class RRTStarPlanner(Planner):
    """
    RRT star path planner

    Args:
        robot(SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
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
        self_collision_manager=None,
        obstacle_collision_manager=None,
        delta_distance=0.5,
        epsilon=0.2,
        max_iter=3000,
        gamma_RRT_star=300, # At least gamma_RRT > delta_distance,
        dimension=7,
        n_step=10
    ):
        super(RRTStarPlanner, self).__init__(
            robot, self_collision_manager, 
            obstacle_collision_manager,
            dimension
        )
        self.delta_dis = delta_distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gamma_RRTs = gamma_RRT_star
        
        self._cur_qpos = None
        self._goal_pose = None
        self.T = None
        self.cost = None

        self.arm = None
        self.dimension = dimension
        self.n_step = n_step
        self.eef_name = self.robot.eef_name

        super()._setup_q_limits()
        super()._setup_eef_name()

    def __repr__(self):
        return 'pykin.planners.rrt_star_planner.{}()'.format(type(self).__name__)
        
    def get_path_in_joinst_space(self, cur_q, goal_pose, resolution=1):
        """
        Get path in joint space

        Returns:
            path(list) : result path (from start joints to goal joints)
        """
        self._cur_qpos = super()._change_types(cur_q)
        self._goal_pose = super()._change_types(goal_pose)

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

            q_paths = None
            self.T = Tree()
            self.cost = {}

            self.T.add_vertex(self._cur_qpos)
            self.cost[0] = 0

            for step in range(self.max_iter):
                if step % 300 == 0 and step !=0:
                    logger.info(f"iter : {step}")
                    
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
                        q_paths = self.find_path(self.T)

            if q_paths is not None:
                logger.info(f"Generate Path Successfully!!")  
                break 

            if cnt > total_cnt:
                logger.error(f"Failed Generate Path.. The number of retries of {cnt} exceeded")
                break

            logger.error(f"Failed Generate Path..")
            print(f"{sc.BOLD}Retry Generate Path, the number of retries is {cnt}/{total_cnt} {sc.ENDC}\n")
        
        result_q_paths = []
        if q_paths is not None:
            for step, joint in enumerate(q_paths):
                if step % round(1/resolution) == 0 or step == len(q_paths)-1:
                    result_q_paths.append(joint)

            interpolate_path = []
            interpolate_paths = []
            
            for i in range(len(result_q_paths)-1):
                interpolate_path = [path.tolist() for path in self._get_linear_path(result_q_paths[i], result_q_paths[i+1])]
                interpolate_paths.extend(interpolate_path)

        return interpolate_paths, result_q_paths

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
        path.append(self._cur_qpos)

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

    def _get_linear_path(self, init_pose, goal_pose):
        for step in range(1, self.n_step + 1):
            delta_t = step / self.n_step
            qpos = get_linear_interpoation(init_pose, goal_pose, delta_t)
            yield qpos