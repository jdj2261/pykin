import sys
import math
import numpy as np

import os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.robots.single_arm import SingleArm

from pykin.planners.planner import Planner
from pykin.kinematics.transform import Transform
from pykin.utils.fcl_utils import FclManager
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import get_robot_geom, limit_joints
from pykin.utils.transform_utils import get_homogeneous_matrix
from pykin.utils.error_utils import NotFoundError

class Tree:
    """
    Tree
    """
    def __init__(self):
        self.vertices = []
        self.edges = []

    def add_vertex(self, q_joints):
        self.vertices.append(q_joints)

    def add_edge(self, q_joints_idx):
        self.edges.append(q_joints_idx)


class Environment:
    """
    Environment (Map, Obstacles)
    """
    def __init__(
        self, 
        x_min, 
        y_min, 
        z_min,
        x_max, 
        y_max,
        z_max
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.obstacles = []

    def add_obstacle(self, obj):
        self.obstacles.extend(obj)


class RRTStarPlanner(Planner):
    """
    RRT path planning
    """
    def __init__(
        self, 
        robot,
        obstacles=[],
        current_q=None,
        goal_q=None,
        delta_distance=0.5,
        epsilon=0.1,
        max_iter=3000,
        gamma_RRT_star=300, # At least gamma_RRT > delta_distance,
        fcl_manager=None
    ):
        self.robot = robot
        self.obstacles = obstacles
        self.cur_q = current_q
        self.goal_q  = goal_q
        self.delta_dis = delta_distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gamma_RRTs = gamma_RRT_star
        self.fcl_manager = fcl_manager

        self.arm = None
        self.dimension = self.robot.dof
        self.eef_name = self.robot.eef_name

        self.T = Tree()
        self.cost = {}

        self._setup_fcl_manager()

    def _setup_fcl_manager(self):
        for link, transformation in self.robot.init_transformations.items():
            name, gtype, gparam = get_robot_geom(self.robot.links[link])
            transform = transformation.homogeneous_matrix
            self.fcl_manager.add_object(name, gtype, gparam, transform)
    def setup_start_goal_joint(self, current_q, goal_q, arm=None):
        self.cur_q = current_q
        self.goal_q  = goal_q
        self.arm = arm
        self.dimension = len(current_q)

        self._get_q_limits()
        self._get_eef_name()

    def generate_path(self):
        if self.cur_q.all() or self.goal_q.all() is None:
            raise NotFoundError("Make sure set current or goal joints..")

        path = None
        self.T = Tree()
        self.cost = {}
        self.T.add_vertex(self.cur_q)
        self.cost[0] = 0



        for k in range(self.max_iter):
            rand_q = self.random_state()
            nearest_q, nearest_idx = self.nearest_neighbor(rand_q, self.T)
            new_q = self.new_state(nearest_q, rand_q)
   
            # if k % 500 == 0:
            #     print(f"iter : {k}")

            if self.collision_free(new_q) and self.q_in_limits(new_q):
                neighbor_indexes = self.find_near_neighbor(new_q)                
                min_cost = self.get_new_cost(nearest_idx, nearest_q, new_q)
                min_cost, nearest_idx = self.get_minimum_cost(neighbor_indexes, new_q, min_cost, nearest_idx)
 
                self.T.add_vertex(new_q)
                new_idx = len(self.T.vertices) - 1
                self.cost[new_idx] = min_cost
                self.T.add_edge([nearest_idx, new_idx])

                self.rewire(neighbor_indexes, new_q, new_idx)

                if self.reach_to_goal(new_q):
                    path = self.find_path(self.T)

        return path

    def random_state(self):
        q_outs = np.zeros(self.dimension)
        if np.random.random() > self.epsilon:
            for i, (q_min, q_max) in enumerate(zip(self.q_limits_lower, self.q_limits_upper)):
                q_outs[i] = np.random.uniform(q_min, q_max)
        else:
            q_outs = self.goal_q
        return q_outs

    def _get_q_limits(self):
        if self.arm is not None:
            self.q_limits_lower = self.robot.joint_limits_lower[self.arm]
            self.q_limits_upper = self.robot.joint_limits_upper[self.arm]
        else:
            self.q_limits_lower = self.robot.joint_limits_lower
            self.q_limits_upper = self.robot.joint_limits_upper

    def _get_eef_name(self):
        if self.arm is not None:
            self.eef_name = self.robot.eef_name[self.arm]

    def q_in_limits(self, q_in):
        return np.all([q_in >= self.q_limits_lower, q_in <= self.q_limits_upper])

    def get_eef_pos(self, q_in):
        if self.arm is not None:
            eef_pos = self.robot.forward_kin(q_in, self.robot.desired_frames[self.arm])[self.eef_name].pos
        else:
            eef_pos = self.robot.forward_kin(q_in)[self.eef_name].pos
        return eef_pos

    def get_eef_pose(self, q_in):
        if self.arm is not None:
            eef_pose = self.robot.forward_kin(q_in, self.robot.desired_frames[self.arm])[self.eef_name].homogeneous_matrix
        else:
            eef_pose = self.robot.forward_kin(q_in)[self.eef_name].homogeneous_matrix
        return eef_pose

    def get_transformations(self, q_in):
        if self.arm is not None:
            transformations = self.robot.forward_kin(q_in, self.robot.desired_frames[self.arm])
        else:
            transformations = self.robot.forward_kin(q_in)
        return transformations

    def nearest_neighbor(self, random_q, tree):
        distances = [self.distance(random_q, vertex) for vertex in tree.vertices]
        nearest_idx = np.argmin(distances)
        nearest_point = tree.vertices[nearest_idx]
        return nearest_point, nearest_idx

    def distance(self, pointA, pointB):
        return np.linalg.norm(pointB - pointA)

    def new_state(self, nearest_q, random_q):
        if np.equal(nearest_q, random_q).all():
            return nearest_q

        vector = random_q - nearest_q
        dist = self.distance(random_q, nearest_q)
        step = min(self.delta_dis, dist)
        unit_vector = vector / dist
        new_q = nearest_q + unit_vector * step

        return new_q

    def collision_free(self, new_q):
        transformations = self.get_transformations(new_q)
        for link, transformations in transformations.items():
            if link in self.fcl_manager._objs:
                transform = transformations.homogeneous_matrix
                self.fcl_manager.set_transform(name=link, transform=transform)

        is_collision = self.fcl_manager.collision_check(return_names=False, return_data=False)
        return False if is_collision else True

    def find_near_neighbor(self, q):
        card_V = len(self.T.vertices) + 1
        r = self.gamma_RRTs * ((math.log(card_V) / card_V) ** (1/self.dimension))
        search_radius = min(r, self.delta_dis)
        dist_list = [self.distance(vertex, q) for vertex in self.T.vertices]
                                                   
        near_indexes = []
        for idx, dist in enumerate(dist_list):
            if dist <= search_radius and self.collision_free(q):
                near_indexes.append(idx)

        return near_indexes

    def get_new_cost(self, idx, A, B):
        cost = self.cost[idx] + self.distance(A, B)
        return cost

    def get_minimum_cost(self, neighbor_indexes, new_q, min_cost, nearest_idx):
        for i in neighbor_indexes:
            new_cost = self.get_new_cost(i, new_q, self.T.vertices[i])

            if new_cost < min_cost and self.collision_free(new_q):
                min_cost = new_cost
                nearest_idx = i

        return min_cost, nearest_idx

    def rewire(self, neighbor_indexes, new_q, new_idx):
        for i in neighbor_indexes:
            no_collision = self.collision_free(self.T.vertices[i])
            new_cost = self.get_new_cost(new_idx, new_q, self.T.vertices[i])

            if no_collision and new_cost < self.cost[i]:
                self.cost[i] = new_cost
                self.T.edges[i-1][0] = new_idx

    def reach_to_goal(self, point):
        dist = self.distance(point, self.goal_q)
        if dist <= 0.02:
            return True
        return False

    def find_path(self, tree):
        path = [self.goal_q]
        goal_idx = tree.edges[-1][1]
 
        while goal_idx != 0:
            path.append(tree.vertices[goal_idx])
            parent_idx = tree.edges[goal_idx-1][0]
            goal_idx = parent_idx
        path.append(self.cur_q)

        return path[::-1]

    def get_rrt_tree(self):
        vertices = []
        for edge in self.T.edges:
            from_node = self.T.vertices[edge[0]]
            goal_node = self.T.vertices[edge[1]]
            vertices.append((from_node, goal_node))
        return vertices


if __name__ == "__main__":


    file_path = '../../asset/urdf/baxter/baxter.urdf'
    if len(sys.argv) > 1:
        robot_name = sys.argv[1]
        file_path = '../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'

    robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

    robot.setup_link_name("base", "right_wrist")
    robot.setup_link_name("base", "left_wrist")

    # set target joints angle
    head_thetas =  np.zeros(1)
    right_arm_thetas = np.array([-np.pi/4 , 0, 0, -np.pi/4, 0 , 0 ,0])
    left_arm_thetas = np.array([np.pi/4 , 0, -np.pi/4, 0, 0 , 0 ,0])

    thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))
    target_transformations = robot.forward_kin(thetas)

    init_q_space = { "right": np.zeros(7), 
                    "left" : np.zeros(7)}

    target_pose = { "right": robot.eef_pose["right"], 
                    "left" : robot.eef_pose["left"]}

    target_q_space = robot.inverse_kin(
        np.random.randn(7), 
        target_pose, 
        method="LM", 
        maxIter=100)

    # robot.remove_desired_frames()
    planner = RRTStarPlanner(
        robot=robot,
        obstacles=[],
        delta_distance=1,
        gamma_RRT_star=10,
        epsilon=0.2, 
        max_iter=100,
        fcl_manager=FclManager()
    )


    trajectory_pos = []
    path = {}
    trajectories = []
    for arm in robot.arms:
        planner.setup_start_goal_joint(init_q_space[arm], target_q_space[arm], arm)
        path[arm] = planner.generate_path()

    from itertools import zip_longest

    if any(value is None for value in path.values()):
        print("Not created trajectories..")
    else:

        current_q_space = { "right": path["right"][-1], "left" : path["left"][-1]}
        trajectory_joints = list(zip_longest(np.array(path["right"]), np.array(path["left"])))
        print(f"target : {target_q_space}")
        print(f"path: {current_q_space}")

        for i, (right_joint, left_joint) in enumerate(trajectory_joints):

            if right_joint is None:
                right_joint = last_right_joint
            if left_joint is None:
                left_joint = last_left_joint

            last_right_joint = right_joint
            last_left_joint = left_joint

            current_joint = np.concatenate((head_thetas, left_joint, right_joint)) 
            transformations = robot.forward_kin(current_joint)
            trajectory_pos.append(transformations)
    
        plt.plot_animation(robot, trajectory_pos, interval=100, repeat=False)