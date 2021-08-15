
import os, sys
import numpy as np
from collections import OrderedDict
import fcl
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)
from pykin.robot import Robot
from pykin.kinematics import transform as tf
from pykin.utils.shell_color import ShellColors as scolors
from pykin.utils import plot as plt

class Collision:
    
    def __init__(self, robot=None, obj=None, fk:dict=None):
        self.robot = robot
        self.obj = []
        self.link_type = OrderedDict()
        self.cylinder = OrderedDict()
        self.box = OrderedDict()
        self.sphere = OrderedDict()
        self.mesh = OrderedDict()

        if obj is not None:
            self.obj.append(obj)

        if robot is not None:
            self.get_link_type()

        if fk is not None:
            self.fk = fk

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info:
            {list(self.link_type.values())}"""
        else:
            return f"""Object Collision Info:
            {self._box}"""

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        self._box = box
        if box.keys():
            if len(self.obj) != 0:
                # TODO
                assert self.obj[0].keys() != box.keys(
                ), f"Duplicate name. please check again"
            self.obj.append(self._box)

    def get_link_type(self):
        if self.robot.desired_frame is not None:
            for desired_frame in self.robot.desired_frame:
                self._link_type_check(frame=desired_frame)
        else:
            self._link_type_check(robot=self.robot)

        self._append_link_type()

    def _link_type_check(self, robot=None, frame=None):
        if robot is not None:
            for info in robot.tree.links.values():
                self.link_type[info.name] = info
                
        if frame is not None:
            if frame.link.dtype in ['box', 'sphere', 'cylinder', 'mesh']:
                self.link_type[frame.link.name] = frame.link

    def _append_link_type(self):
        for link, info in self.link_type.items():
            if info.dtype == 'box':
                self.box[link] = info
            if info.dtype == 'sphere':
                self.sphere[link] = info
            if info.dtype == 'cylinder':
                self.cylinder[link] = info
            if info.dtype == 'mesh':
                self.mesh[link] = info

    def pairwise_collsion_check(self):
        pass

    def pairwise_distance_check(self):
        pass

    def continous_collision_check(self):
        pass

    def plot(self, name="Test"):
        _, ax = plt.init_3d_figure(name)

        for obj_type, info in obj.items():
            if self.robot is not None:
                if info.dtype == 'box':
                    A2B = np.eye(4)
                    plt.plot_box(ax=ax, size=info.size, A2B=A2B, alpha=0.1, color='k')
            else:
                if obj_type == 'box':
                    A2B = np.eye(4)
                    plt.plot_box(ax=ax, size=info.size, A2B=A2B, alpha=0.1, color='k')
            
            # if info.dtype == 'sphere':
            #     self.sphere[link] = info
            # if info.dtype == 'cylinder':
            #     self.cylinder[link] = info
            # if info.dtype == 'mesh':
            #     self.mesh[link] = info

if __name__ == "__main__":
    file_path = '../../asset/urdf/baxter/baxter.urdf'
    robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
    robot.set_desired_tree("base", "left_gripper")

    box_size = (1.0, 2.0, 3.0)
    box1 = {'box1': box_size}
    box2 = {'box2': box_size}
    # box = fcl.Box(box_size)

    col = Collision()
    col.box = box1
    col.box = box2
    print(col.obj)
    # col.plot()

    # plt.show_figure()
