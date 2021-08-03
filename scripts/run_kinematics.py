#! /usr/bin/python3

import sys
import os
import argparse
# pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
# sys.path.append(pykin_path)

from pykin.robot import Robot
from pprint import pprint
from pykin.utils import plot as plt

if __name__ == "__main__":
    file_path = "../asset/urdf/panda.urdf"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    robot = Robot(file_path)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(robot)

    head_thetas = [0.0]
    right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
    left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

    th = head_thetas + left_arm_thetas + right_arm_thetas
    fk = robot.forward_kinematics(th)
    _, ax = plt.init_3d_figure()
    plt.plot_robot(robot, fk, ax, file_name)
    ax.legend()
    plt.show_figure()
