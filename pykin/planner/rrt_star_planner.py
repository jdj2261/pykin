import sys
import math
import numpy as np
import random

import os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robot import Robot
from pykin.kinematics.transform import Transform
from pykin.utils.fcl_utils import FclManager
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import get_robot_geom, limit_joints
from pykin.utils.transform_utils import get_homogeneous_matrix


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


class RRTStar:
    """
    RRT path planning
    """
    def __init__(
        self, 
        robot,
        obstacles,
        current_q,
        goal_q,
        delta_distance=0.5,
        epsilon=0.1,
        max_iter=3000,
        gamma_RRT_star=300, # At least gamma_RRT > delta_distance,
        fcl_manager=None
    ):
        self.robot = obstacles
        self.obstacles = obstacles
        self.cur_q = current_q
        self.goal_q  = goal_q
        self.delta_dis = delta_distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gamma_RRTs = gamma_RRT_star
        self.fcl_manager = fcl_manager

        self.dimension = len(self.eef_pos)
        self.T = Tree()
        self.cost = {}

    def generate_path(self):
        path = None
        self.T.add_vertex(self.eef_pos)
        self.cost[0] = 0

        for k in range(self.max_iter):
            rand_point = self.random_state()
            nearest_point, nearest_idx = self.nearest_neighbor(rand_point, self.T)
            new_point = self.new_state(nearest_point, rand_point)

            if k % 500 == 0:
                print(f"iter : {k}")

            if self.collision_free(nearest_point, new_point):
                neighbor_indexes = self.find_near_neighbor(new_point)                
                min_cost = self.get_new_cost(nearest_idx, nearest_point, new_point)
                min_cost, nearest_idx = self.get_minimum_cost(neighbor_indexes, new_point, min_cost, nearest_idx)
 
                self.T.add_vertex(new_point)
                new_idx = len(self.T.vertices) - 1
                self.cost[new_idx] = min_cost
                self.T.add_edge([nearest_idx, new_idx])

                self.rewire(neighbor_indexes, new_point, new_idx)

                if self.reach_to_goal(new_point):
                    path = self.find_path(self.T)

        return path

    # TODO
    def random_state(self):
        if np.random.random() > self.epsilon:
            point = np.array([np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1)]) 
        else:
            point = self.goal_pos
        return point


    # TODO
    # make joint_lower_limits
    def check_joint_limits(self, joints):
        for i, joint in enumerate(joints):
            if (joint < self.robot.joint_lower_limits[i] or \
                joint > self.robot.joint_upper_limits[i]):
                return True
        return False

    def nearest_neighbor(self, random_point, tree):
        distances = [self.distance(random_point, point) 
                     for point in tree.vertices]
        nearest_idx = np.argmin(distances)
        nearest_point = tree.vertices[nearest_idx]
        return nearest_point, nearest_idx

    def distance(self, pointA, pointB):
        return np.linalg.norm(pointB - pointA)

    def new_state(self, nearest_point, random_point):
        if np.equal(nearest_point, random_point).all():
            return nearest_point

        vector = random_point - nearest_point
        dist = self.distance(random_point, nearest_point)
        step = min(self.delta_dis, dist)
        unit_vector = vector / dist
        new_point = nearest_point + unit_vector * step

        return new_point

    # TODO
    # use fcl lib
    # fcl set_transform
    def collision_free(self, pointA, pointB):
        self.fcl_manager.set_transform
        is_collision = self.fcl_manager.collision_check(return_names=False, return_data=False)
        if is_collision:
            return False
        return True

    def find_near_neighbor(self, point):
        card_V = len(self.T.vertices) + 1
        r = self.gamma_RRTs * ((math.log(card_V) / card_V) ** (1/self.dimension))
        search_radius = min(r, self.delta_dis)
        dist_list = [self.distance(vertex, point) for vertex in self.T.vertices]
                                                   
        near_indexes = []
        for idx, dist in enumerate(dist_list):
            if dist <= search_radius and self.collision_free(point, self.T.vertices[idx]):
                near_indexes.append(idx)

        return near_indexes

    def get_new_cost(self, idx, A, B):
        cost = self.cost[idx] + self.distance(A, B)
        return cost

    def get_minimum_cost(self, neighbor_indexes, new_point, min_cost, nearest_idx):
        for i in neighbor_indexes:
            new_cost = self.get_new_cost(i, new_point, self.T.vertices[i])

            if new_cost < min_cost and self.collision_free(new_point, self.T.vertices[i]):
                min_cost = new_cost
                nearest_idx = i

        return min_cost, nearest_idx

    def rewire(self, neighbor_indexes, new_point, new_idx):
        for i in neighbor_indexes:
            no_collision = self.collision_free(new_point, self.T.vertices[i])
            new_cost = self.get_new_cost(new_idx, new_point, self.T.vertices[i])

            if no_collision and new_cost < self.cost[i]:
                self.cost[i] = new_cost
                self.T.edges[i-1][0] = new_idx

    def reach_to_goal(self, point):
        dist = self.distance(point, self.goal_pos)
        if dist <= 0.5:
            return True
        return False

    def find_path(self, tree):
        path = [self.goal_pos]
        goal_idx = tree.edges[-1][1]
 
        while goal_idx != 0:
            path.append(tree.vertices[goal_idx])
            parent_idx = tree.edges[goal_idx-1][0]
            goal_idx = parent_idx
        path.append(self.eef_pos)

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

    eef_name = "left_wrist"

    if len(sys.argv) > 1:
        robot_name = sys.argv[1]
        file_path = '../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'

    fig, ax = plt.init_3d_figure("URDF")
    robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

    right_joint_idxes = []
    left_joint_idxes = []
    for i, joint in enumerate(robot.joints.values()):
        if joint.dtype == 'fixed':
            continue
        if "right" in joint.name:
            right_joint_idxes.append(i)
        if "left" in joint.name:
            left_joint_idxes.append(i)
    print(right_joint_idxes, left_joint_idxes)

    for r_idx in right_joint_idxes:
        print(list(robot.links.values())[r_idx]) 
        print(list(robot.joints.values())[r_idx]) 
    

    # robot_eef_pos = robot.transformations.get(eef_name).pos

    # ###### IK ######
    # left_arm_thetas = np.array([np.pi/2, 0, 0, 0, 0, 0, 0])
    # init_left_thetas = np.random.randn(7)

    # # Set desired frame (root, end)
    # robot.set_desired_frame("base", eef_name)
    # left_arm_fk = robot.kin.forward_kinematics(left_arm_thetas)
    # goal_eef_pos = left_arm_fk[eef_name].pos

    # target_l_pose = np.concatenate((left_arm_fk["left_wrist"].pos, left_arm_fk["left_wrist"].rot))
    # # Left's arm IK solution by LM
    # ik_left_LM_result, _= robot.kin.inverse_kinematics(init_left_thetas, target_l_pose, method="LM", maxIter=100)
    # transformations = robot.kin.forward_kinematics(ik_left_LM_result)

    # print(robot_eef_pos, goal_eef_pos)


    # fcl_manager = FclManager()

    # spheres = []
    # radius = 0.1
    # for i in range(5):
    #     x = np.random.uniform(-1, 1)
    #     y = np.random.uniform(-1, 1)
    #     z = np.random.uniform(-1, 1)

    #     name = "obstacle_" + str(i)
    #     obs_transform = get_homogeneous_matrix(position=np.array([x, y, z]))

    #     fcl_manager.add_object( name, 
    #                             gtype="sphere", 
    #                             gparam=radius,
    #                             transform=obs_transform)
    #     spheres.append((x, y, z, radius))

    # for link, transformation in transformations.items():
    #     name, gtype, gparam = get_robot_geom(robot.links[link])
    #     transform = transformation.homogeneous_matrix
    #     fcl_manager.add_object(name, gtype, gparam, transform)

    # result, objs_in_collision, contact_data = fcl_manager.collision_check(return_names=True, return_data=True)
    # print(result, objs_in_collision, contact_data)


    # planner = RRTStar( current_q=robot_eef_pos, 
    #                    goal_q=goal_eef_pos, 
    #                    delta_distance=0.02,
    #                    gamma_RRT_star=0.5,
    #                    epsilon=0.2, 
    #                    max_iter=2000,
    #                    fcl_manager=fcl_manager)

    # path = planner.generate_path()
    # vertices = planner.get_rrt_tree()

    # # Plot
    # for sp_x, sp_y, sp_z, sp_r in spheres:
    #     sp_radius = sp_r
    #     sp_pos = np.array([sp_x, sp_y, sp_z])
    #     plt.plot_sphere(ax, radius=radius, p=sp_pos, alpha=0.2, color="k")

    # # plt.plot_rrt_vertices(vertices, ax)
    # # plt.plot_path_planner(path, ax)
    # plt.plot_robot(robot, 
    #             transformations,
    #             ax=ax, 
    #             name="baxter_test",
    #             visible_visual=False, 
    #             visible_collision=True, 
    #             mesh_path='../../asset/urdf/baxter/')
    # ax.legend()
    # plt.show_figure()