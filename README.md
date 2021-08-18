# pykin
[![PyPI version](https://badge.fury.io/py/pykin.svg)](https://badge.fury.io/py/pykin)  [![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Python Interface for the Robot Kinematics Library

This library has been created simply by referring to [ikpy](https://github.com/Phylliade/ikpy.git).

## Features

- Pure python library
- Support only URDF file
- Compute Forward, Inverse Kinematics and Jacobian
- Compute Collision checkinkg
- Plot Robot Kinematic Chain
- Show Robot Mesh

## Installation

- You need a package to do collision checking.

~~~
pip install pykin
~~~



~~~
git clone https://github.com/jdj2261/pykin.git
~~~

## Quick Start

- Robot Info

  ~~~python
  import pykin.robot
  from pykin.robot import Robot
  
  file_path = '../asset/urdf/baxter.urdf'
  
  robot = Robot(file_path)
  robot.show_robot_info()
  
  print(robot.joints)
  print(robot.links)
  print(robot.tree.root)
  print(robot.num_links)
  print(robot.num_active_joints)
  print(robot.get_active_joint_names) 
  ~~~

- Forward Kinematics

  ~~~python
  import sys
  import os
  import numpy as np
  from pprint import pprint
  from pykin import robot
  from pykin.robot import Robot
  from pykin.kinematics import transform as tf
  from pykin.utils import plot as plt
  file_path = '../asset/urdf/baxter.urdf'
  
  robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
  
  # baxter_example
  head_thetas = [0.0]
  right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
  left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
  
  thetas = head_thetas + right_arm_thetas + left_arm_thetas
  fk = robot.forward_kinematics(thetas)
  
  """
  If you want to know transformations of all links,
  you don't have to write get_desired_tree and desired_frame.
  """
  pprint(fk)
  for link, T in fk.items():
      print(f"link: {link}, pose:{np.concatenate((T.pos, T.rot))} ")
  ~~~

- Inverse Kinematics

  ~~~python
  import sys
  import os
  import numpy as np
  from pprint import pprint
  from pykin import robot
  from pykin.robot import Robot
  from pykin.kinematics import transform as tf
  from pykin.utils import plot as plt
  
  file_path = '../asset/urdf/baxter.urdf'
  
  robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
  
  # baxter_example
  head_thetas = [0.0]
  right_arm_thetas = [-np.pi/4, 0, 0, 0, 0, 0, 0]
  left_arm_thetas = [np.pi/4, 0, 0, 0, 0, 0, 0]
  
  init_right_thetas = [0, 0, 0, 0, 0, 0, 0]
  init_left_thetas = [0, 0, 0, 0, 0, 0, 0]
  
  robot.set_desired_tree("base", "right_wrist")
  right_arm_fk = robot.forward_kinematics(right_arm_thetas)
  target_r_pose = np.concatenate(
      (right_arm_fk["right_wrist"].pos, right_arm_fk["right_wrist"].rot))
  ik_right_result = robot.inverse_kinematics(
      init_right_thetas, target_r_pose, method="numerical")
  
  robot.desired_frame = None
  thetas = head_thetas + ik_right_result + left_arm_thetas
  fk = robot.forward_kinematics(thetas)
  
  _, ax = plt.init_3d_figure()
  plt.plot_robot(robot, fk, ax, "baxter")
  ax.legend()
  plt.show_figure()
  ~~~

- Result

  ![baxter img](img/baxter.png)

## Inverse Kinematics 

You can see an example of IK by running the command below.

~~~shell
$ cd pykin/tests
$ python robot_ik_baxter_test.py
~~~

- **Forward Kinematics**

  <img src="img/FK.png" height="400">

- **IK Newton Raphson method**

  <img src="img/NR.png" height="400">

- **IK Levenberg-Marquardt method**

  <img src="img/LM.png" height="400">



**It can be seen that the LM method is faster and more accurate than the NR method when using IK.**

