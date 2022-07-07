import pprint
import string
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
        self.goal_object = 'goal_box'

        # TODO
        if self.bench_num == 1:
            if self.benchmark_config[self.bench_num].get("goal_object"):
                self.goal_object = self.benchmark_config[self.bench_num]["goal_object"]
            self.goal_stacked_num:int = self.benchmark_config[self.bench_num]["stack_num"]
            self.alphabet_list:list = list(string.ascii_uppercase)[:self.goal_stacked_num]
            self.goal_boxes:list = [alphabet + '_box' for alphabet in self.alphabet_list]
            self.stacked_box_num = 0
            self.success_stacked_box_num = 0

        if self.bench_num == 2 and self.benchmark_config[self.bench_num] is not None:
            if self.benchmark_config[self.bench_num].get("goal_object"):
                self.goal_object = self.benchmark_config[self.bench_num]["goal_object"]

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
            return self.check_terminal_state_bench_2()
        if self.bench_num == 3:
            pass
        if self.bench_num == 4:
            pass

    def check_terminal_state_bench_1(self):
        is_success = self.check_success_stacked_bench_1(is_terminal=True)
        if is_success and self.stacked_box_num == self.goal_stacked_num:
            return True
        return False

    def check_success_stacked_bench_1(self, is_terminal=False):
        is_success = False

        stacked_boxes = self.get_objs_chain_list_from_bottom(self.goal_object)[1:]
        stacked_box_num = len(stacked_boxes)
        self.stacked_box_num = stacked_box_num

        if stacked_box_num <= self.goal_stacked_num:
            goal_stacked_boxes = self.goal_boxes[:stacked_box_num]
            if stacked_boxes == goal_stacked_boxes:
                is_success = True
                if not is_terminal:
                    self.success_stacked_box_num = stacked_box_num
                
        return is_success

    def get_objs_chain_list_from_bottom(self, bottom_obj):
        support_objs:list = self.logical_states[bottom_obj].get(self.logical_state.support)
        if not support_objs:
            return [bottom_obj]
        else:
            upper_obj = support_objs[0].name
            return [bottom_obj] + self.get_objs_chain_list_from_bottom(upper_obj)

    def check_terminal_state_bench_2(self):
        is_success = False
        print(self.robot.gripper.attached_obj_name, self.goal_object)
        if self.robot.gripper.attached_obj_name == self.goal_object:
            is_success = True
        return is_success