import pprint
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
        self.benchmark_config = benchmark
        self.bench_num = list(self.benchmark_config.keys())[0]
        self.stacked_obj_num = list(self.benchmark_config[self.bench_num].values())[0]
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
        if self.pick_obj_name is None:
            return False

        objs_chain_list = deepcopy(self.get_objs_chain_list(self.pick_obj_name, []))
        objs_chain_list.pop(-1)

        sorted_chain_list = sorted(objs_chain_list, reverse=True)
        
        if "goal_box" in sorted_chain_list:
            objs_chain_list.remove("goal_box")
            sorted_chain_list.remove("goal_box")
        
            if len(objs_chain_list) == self.stacked_obj_num:
                if objs_chain_list == sorted_chain_list:
                    return True
        return False
        
    def get_objs_chain_list(self, held_obj_name, obj_chain=[]):
        if held_obj_name not in self.objs:
            raise ValueError(f"Not found {held_obj_name} in this scene")
        
        if self.logical_state.on in list(self.logical_states[held_obj_name].keys()):
            support_obj = self.logical_states[held_obj_name][self.logical_state.on]
            obj_chain.append(support_obj.name)
            if support_obj.name != "table":
                self.get_objs_chain_list(support_obj.name, obj_chain)
        
        return [held_obj_name] + obj_chain