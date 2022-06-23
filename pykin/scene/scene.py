import pprint
import numpy as np
import string

from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass

from pykin.robots.single_arm import SingleArm
from pykin.utils.kin_utils import ShellColors as sc

@dataclass
class State:
    on = 'on'
    support = 'support'
    static = 'static'
    held = 'held'
    holding = 'holding'

class Scene:
    def __init__(self, benchmark:dict):

        self.benchmark_config:int = benchmark
        self.bench_num:int = list(self.benchmark_config.keys())[0]
        self.goal_stacked_num:int = self.benchmark_config[self.bench_num]["stack_num"]
        self.alphabet_list:list = list(string.ascii_uppercase)[:self.goal_stacked_num]
        self.goal_box_list:list = [alphabet + '_box' for alphabet in self.alphabet_list]
        self.success_cnt = 0

        self.objs:dict = {}
        self.robot:SingleArm = None
        self.logical_states:OrderedDict = OrderedDict()
        self.logical_state:State = State
        
        self.grasp_poses = None
        self.release_poses = None
        
        self.pick_obj_name = None
        self.place_obj_name = None
        self.pick_obj_default_pose = None
        
    def show_scene_info(self):
        print(f"*"*23 + f" {sc.OKGREEN}Scene{sc.ENDC} "+ f"*"*23)
        pprint.pprint(self.objs)
        if self.robot:
            print(self.robot.robot_name, self.robot.offset)
            if self.robot.has_gripper:
                print(self.robot.gripper.name, self.robot.gripper.get_gripper_pose())
        print(f"*"*63 + "\n")

    def show_logical_states(self):
        print(f"*"*23 + f" {sc.OKGREEN}Logical States{sc.ENDC} "+ f"*"*23)
        pprint.pprint(self.logical_states)
        print(f"*"*63 + "\n")

    def update_logical_states(self):
        for object_name, logical_state in self.logical_states.items():
            if logical_state.get(State.on):
                if not self.logical_states[logical_state[State.on].name].get(State.support):
                    self.logical_states[logical_state[State.on].name][State.support] = []
                                
                if self.objs[object_name] not in list(self.logical_states[logical_state[State.on].name].get(State.support)):
                    self.logical_states[logical_state[State.on].name][State.support].append(self.objs[object_name])

            if logical_state.get(State.support) is not None and not logical_state.get(State.support):
                self.logical_states[object_name].pop(State.support)

            if logical_state.get(State.holding):
                self.logical_states[logical_state[State.holding].name][State.held] = True

    # Add for MCTS
    def is_terminal_state(self):
        if self.bench_num == 1:
            return self.check_terminal_state_bench_1()
        if self.bench_num == 2:
            pass
        if self.bench_num == 3:
            pass
        if self.bench_num == 4:
            pass

    def check_terminal_state_bench_1(self):
        is_success, stacked_obj_num = self.check_success_stacked_bench_1(is_terminal=True)

        if is_success and stacked_obj_num == self.goal_stacked_num:
            return True
        return False

    def check_success_stacked_bench_1(self, is_terminal=False):
        is_success = False

        stacked_objs = self.get_objs_chain_list_from_bottom("goal_box")[1:]
        stacked_num = len(stacked_objs)

        if stacked_num <= self.goal_stacked_num:
            cur_goal_list = self.goal_box_list[:stacked_num]
            if stacked_objs == cur_goal_list:
                is_success = True
                if not is_terminal:
                    self.success_cnt = stacked_num
        
        return is_success, stacked_num

    def get_objs_chain_list_from_bottom(self, bottom_obj):
        support_objs:list = self.logical_states[bottom_obj].get(self.logical_state.support)
        if not support_objs:
            return [bottom_obj]
        else:
            upper_obj = support_objs[0].name
            return [bottom_obj] + self.get_objs_chain_list_from_bottom(upper_obj)