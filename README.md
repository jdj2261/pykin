# pykin

[![PyPI version](https://badge.fury.io/py/pykin.svg)](https://badge.fury.io/py/pykin)  [![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Python Interface for the Robot Kinematics Library [pykin](https://jdj2261.github.io/pykin/)

Even motion planning is supported.

This library has been created simply by referring to [ikpy](https://github.com/Phylliade/ikpy.git).

## Features

- Pure python library
- Support only URDF file
- Compute Forward, Inverse Kinematics and Jacobian
- There are two ways to find the IK solution, referring to the [Introduction to Humanoid Robotics book](https://link.springer.com/book/10.1007/978-3-642-54536-8).
- Compute Collision checkinkg
- Planning (RRT Star motion planning)
- Plot Robot Kinematic Chain and Robot Mesh (STL file)

## Installation

### Requirements

You need a [python-fcl](https://github.com/BerkeleyAutomation/python-fcl) package to do object collision checking.

- For Ubuntu, using  `apt`

  `sudo apt install liboctomap-dev`

  `sudo apt install libfcl-dev`
- For Mac, First, Download the source and build it.

  - octomap

    `git clone https://github.com/OctoMap/octomap.git`

    ~~~
    $ cd octomap
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ make install
    ~~~
  - fcl

    `git clone https://github.com/flexible-collision-library/fcl.git`

    Since python-fcl uses version 0.5.0 of fcl, checkout with tag 0.5.0

    ~~~
    $ cd fcl
    $ git checkout 0.5.0
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ make install
    ~~~

If the above installation is complete

~~~
pip install python-fcl
~~~

### Install Pykin

~~~
pip install pykin
~~~

When git clone, use the --recurse-submodules option.

The download may take a long time due to the large urdf file size.

~~~
git clone --recurse-submodules https://github.com/jdj2261/pykin.git
~~~

If you hadn't done this

~~~
$ git clone https://github.com/jdj2261/pykin.git
$ git submodule update
~~~

## Quick Start

- Robot Info

  You can see 4 example robot information.

  `baxter, iiwa14, panda, and sawyer`

  <details>
    <summary>Code</summary>

  ~~~python
  import sys
  
  file_path = '../asset/urdf/baxter/baxter.urdf'
  
  if len(sys.argv) > 1:
      robot_name = sys.argv[1]
      file_path = '../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
  
  if "baxter" in file_path:
      from pykin.robots.bimanual import Bimanual
      robot = Bimanual(file_path)
  else:
      from pykin.robots.single_arm import SingleArm
      robot = SingleArm(file_path)
  
  robot.show_robot_info()
  ~~~
  
  </details>
- Forward Kinematics

  <details>
    <summary>Code</summary>

  ~~~python
  import numpy as np
  
  from pykin.robots.bimanual import Bimanual
  from pykin.kinematics.transform import Transform
  from pykin.utils import plot_utils as plt
  from pykin.utils.kin_utils import ShellColors as sc
  
  # baxter_example
  file_path = '../../asset/urdf/baxter/baxter.urdf'
  robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
  
  head_thetas = [0.0]
  right_arm_thetas = [np.pi/3, -np.pi/5, -np.pi/2, np.pi/7, 0, np.pi/7 ,0]
  left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
  
  thetas = head_thetas + right_arm_thetas + left_arm_thetas
  fk = robot.forward_kin(thetas)
  
  for link, transform in fk.items():
      print(f"{sc.HEADER}{link}{sc.ENDC}, {transform.rot}, {transform.pos}")
  ~~~
  
  </details>
- Jacobian

  <details>
    <summary>Code</summary>

  ~~~python
  import numpy as np
  
  from pykin.kinematics import transform as tf
  from pykin.robots.bimanual import Bimanual
  from pykin.kinematics import jacobian as jac
  
  file_path = '../asset/urdf/baxter/baxter.urdf'
  robot = Bimanual(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
  
  left_arm_thetas = np.zeros(15)
  robot.setup_link_name("base", "right_wrist")
  robot.setup_link_name("base", "left_wrist")
  
  fk = robot.forward_kin(left_arm_thetas)
  
  J = {}
  for arm in robot.arm_type:
      if robot.eef_name[arm]:
          J[arm] = jac.calc_jacobian(robot.desired_frames[arm], fk, len(np.zeros(7)))
  
  print(J)
  ~~~
  
  </details>
- Inverse Kinematics

  <details>
    <summary>Code</summary>

  ~~~python
  import numpy as np
  
  from pykin.robots.bimanual import Bimanual
  from pykin.kinematics.transform import Transform
  from pykin.utils import plot_utils as plt
  
  file_path = '../../asset/urdf/baxter/baxter.urdf'
  
  robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
  
  visible_collision = True
  visible_visual = False
  
  # set target joints angle
  head_thetas =  np.zeros(1)
  right_arm_thetas = np.array([-np.pi/4 , 0, 0, 0, 0 , 0 ,0])
  left_arm_thetas = np.array([np.pi/4 , 0, 0, 0, 0 , 0 ,0])
  
  thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))
  
  robot.setup_link_name("base", "right_wrist")
  robot.setup_link_name("base", "left_wrist")
  
  #################################################################################
  #                                Set target pose                                #
  #################################################################################
  target_transformations = robot.forward_kin(thetas)
  _, ax = plt.init_3d_figure("Target Pose")
  plt.plot_robot(robot, 
                 ax=ax,
                 transformations=target_transformations,
                 visible_visual=visible_visual, 
                 visible_collision=visible_collision,
                 mesh_path='../../asset/urdf/baxter/')
  
  #################################################################################
  #                                Inverse Kinematics                             #
  #################################################################################
  init_thetas = np.random.randn(7)
  target_pose = { "right": robot.get_eef_pose(target_transformations)["right"], 
                  "left" : robot.get_eef_pose(target_transformations)["left"]}
  
  ik_LM_result = robot.inverse_kin(
      init_thetas, 
      target_pose, 
      method="LM", 
      maxIter=100)
  
  ik_NR_result = robot.inverse_kin(
      init_thetas, 
      target_pose, 
      method="NR", 
      maxIter=100)
  
  print(ik_LM_result, ik_NR_result)
  ~~~
  
  </details>
- Self-Collision Check

  <details>
    <summary>Code</summary>

  ~~~python
  import numpy as np
  
  from pykin.kinematics.transform import Transform
  from pykin.robot import Robot
  
  """
  If you want to check robot's collision, install python-fcl 
  And then, import CollisionManager in collision_utils package
  """
  from pykin.collision.collision_manager import CollisionManager
  from pykin.utils.collision_utils import get_robot_collision_geom
  from pykin.utils import plot_utils as plt
  
  file_path = '../asset/urdf/baxter/baxter.urdf'
  
  robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
  
  head_thetas = np.zeros(1)
  right_arm_thetas = np.array([np.pi, 0, 0, 0, 0, 0, 0])
  left_arm_thetas = np.array([-np.pi, 0, 0, 0, 0, 0, 0])
  
  thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
  transformations = robot.forward_kin(thetas)
  
  # call CollisionManager class
  collision_manager = CollisionManager()
  for link, transformation in transformations.items():
      # get robot link's name and geometry info 
      name, gtype, gparam = get_robot_collision_geom(robot.links[link])
      # get 4x4 size homogeneous transform matrix
      transform = transformation.h_mat
      # add link name, geometry info, transform matrix to collision_manager 
      collision_manager.add_object(name, gtype, gparam, transform)
  
  # you can get collision result, contacted object name, fcl contatct_data
  result, objs_in_collision, contact_data = collision_manager.collision_check(return_names=True, return_data=True)
  
  print(result, objs_in_collision, contact_data)
  
  """
  If you want to check collision check after transform, 
  add the link name and transform matrix to the set_transform function.
  """
  left_arm_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
  thetas = np.hstack((head_thetas, right_arm_thetas, left_arm_thetas))
  transformations = robot.forward_kin(thetas)
  
  for link, transformation in transformations.items():
      name, _, _ = get_robot_collision_geom(robot.links[link])
      transform = transformation.h_mat
      collision_manager.set_transform(name=name, transform=transform)
  
  result, objs_in_collision, contact_data = collision_manager.collision_check(return_names=True, return_data=True)
  print(result, objs_in_collision, contact_data)
  ~~~

  </details>

## Visualization

- urdf

  *You can see visualization using matplotlib.*


|          baxter          |          sawyer          |          iiwa14          |          panda          |
| :-------------------------: | :-------------------------: | :-------------------------: | :-----------------------: |
| ![baxter](img/baxter.png) | ![sawyer](img/sawyer.png) | ![iiwa14](img/iiwa14.png) | ![panda](img/panda.png) |

  <details>
    <summary>Code</summary>

  ~~~python
import sys
file_path = '../../asset/urdf/baxter/baxter.urdf'
if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'

if "baxter" in file_path:
    from pykin.robots.bimanual import Bimanual
    robot = Bimanual(file_path)
else:
    from pykin.robots.single_arm import SingleArm
    robot = SingleArm(file_path)
from pykin.utils import plot_utils as plt


fig, ax = plt.init_3d_figure("URDF")

# For Baxter robots, the name argument to the plot_robot function must be baxter.
plt.plot_robot(robot, 
               ax=ax, 
               visible_visual=False, 
               visible_collision=False, 
               mesh_path='../../asset/urdf/baxter/')
plt.show_figure()
  ~~~
 </details>

- collision

  *You can see collision defined in collision/geometry tags in urdf.*


|               baxter               |               sawyer               |
| :-----------------------------------: | :-----------------------------------: |
| ![baxter](img/baxter_collision.png) | ![baxter](img/sawyer_collision.png) |

  <details>
    <summary>Code</summary>

  ~~~python
import sys

from pykin.robots.bimanual import Bimanual
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/baxter/baxter.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
robot = Bimanual(file_path)

if "baxter" in file_path:
    from pykin.robots.bimanual import Bimanual
    robot = Bimanual(file_path)
else:
    from pykin.robots.single_arm import SingleArm
    robot = SingleArm(file_path)

fig, ax = plt.init_3d_figure("URDF")

"""
Only baxter and sawyer robots can see collisions.
It is not visible unless sphere, cylinder, and box are defined in collision/geometry tags in urdf.
"""
# If visible_collision is True, visualize collision
plt.plot_robot(robot, 
               ax=ax, 
               visible_visual=False, 
               visible_collision=True, 
               mesh_path='../../asset/urdf/baxter/')
ax.legend()
plt.show_figure()
  ~~~
</details>

- mesh

  *You can see  mesh defined in visual/geometry tags in urdf.*
  
  ![baxter](img/all_robot.png)

  <details>
    <summary>Code</summary>

  ~~~python
import sys

file_path = '../../asset/urdf/baxter/baxter.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'

if "baxter" in file_path:
    from pykin.robots.bimanual import Bimanual
    robot = Bimanual(file_path)
else:
    from pykin.robots.single_arm import SingleArm
    robot = SingleArm(file_path)
from pykin.utils import plot_utils as plt
fig, ax = plt.init_3d_figure("URDF")

"""
Only baxter and sawyer robots can see collisions.
It is not visible unless sphere, cylinder, and box are defined in collision/geometry tags in urdf.
"""
# If visible_visual is True, visualize mesh
# and you have to input mesh_path
plt.plot_robot(robot, 
               ax=ax, 
               visible_visual=True, 
               visible_collision=False, 
               mesh_path='../../asset/urdf/'+robot.robot_name+'/')
"""
The mesh file doesn't use matplotlib, 
so it's okay to comment out the line below.
"""
plt.show_figure()
  ~~~
</details>

- Planning

  *You can see an planning animation that visualizes trajectory*

  |                         baxter urdf                          |                       baxter collision                       |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="img/pykin_animation.gif" weight="600" height="300"/> | <img src="img/pykin_animation-2.gif" weight="600" height="300"/> |

  |                       Sawyer collision                       |                         iiwa14 urdf                          |                          panda urdf                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="img/pykin_animation-3.gif" weight="600" height="300"/> | <img src="img/pykin_animation-4.gif" weight="600" height="300"/> | <img src="img/pykin_animation-5.gif" weight="600" height="300"/> |

  

