#! /usr/bin/python3

import sys
import os
import argparse

# import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"../"))

from pykin.robot import Robot
from pprint import pprint
from pykin.utils import plot as plt

if __name__ == "__main__":
    file_name = "../asset/urdf/baxter.urdf"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]

    robot = Robot(file_name)

    print(robot)

    head_thetas = [0.0]
    right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
    left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

    th = head_thetas + left_arm_thetas + right_arm_thetas
    pprint(robot.forward_kinematics(th))
    _, ax = plt.init_3d_figure()
    plt.plot_robot(robot, th, ax, "baxter")
    ax.legend()
    plt.show_figure()
