import pprint
import numpy as np
import string

from itertools import takewhile
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
        self.stacked_obj_num:int = self.benchmark_config[self.bench_num]["stack_num"]
        self.top_box:str = self.benchmark_config[self.bench_num]["top_box"]
        self.alphabet_list = sorted(list(takewhile(lambda x: x <= self.top_box.split('_')[0], list(string.ascii_uppercase))), reverse=True)
        self.goal_box_list = [alphabet + '_box' for alphabet in self.alphabet_list]
        
        self.objs = {}
        self.robot:SingleArm = None
        self.logical_states = OrderedDict()
        self.logical_state = State
        
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
        if self.check_state_bench_1() == self.stacked_obj_num:
            return True
        return False

    def check_state_bench_1(self):
        if self.pick_obj_name is None:
            return 0

        objs_chain_list = deepcopy(self.get_objs_chain_list(self.pick_obj_name, []))
        objs_chain_list.pop(-1)
        success_cnt = 0

        if "goal_box" in objs_chain_list:
            objs_chain_list.remove("goal_box")
            stacked_num = len(objs_chain_list)
            
            objs_chains = np.array(objs_chain_list)[::-1]
            sorted_chains = np.array(self.goal_box_list[len(self.goal_box_list)- stacked_num : ])[::-1]
            
            for i, goal_box in enumerate(sorted_chains):
                if goal_box != objs_chains[i]:
                    break
                success_cnt += 1
            
            if success_cnt == len(sorted_chains):
                return success_cnt

        success_cnt = 0
        return success_cnt

    def get_objs_chain_list(self, held_obj_name, obj_chain=[]):
        if held_obj_name not in self.objs:
            raise ValueError(f"Not found {held_obj_name} in this scene")
        
        if self.logical_state.on in list(self.logical_states[held_obj_name].keys()):
            support_obj = self.logical_states[held_obj_name][self.logical_state.on]
            obj_chain.append(support_obj.name)
            if support_obj.name != "table":
                self.get_objs_chain_list(support_obj.name, obj_chain)
        
        return [held_obj_name] + obj_chain