# pykin

[![PyPI version](https://badge.fury.io/py/pykin.svg)](https://badge.fury.io/py/pykin)  [![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Python Interface for the robot Kinematics library pykin

This library has been created simply by referring to <a href="https://github.com/Phylliade/ikpy.git" target="_blank">ikpy</a>.

## Features

- Pure python library
- Support only URDF file.
- Compute forward, inverse kinematics and jacobian, referred to the [Introduction to Humanoid Robotics book](https://link.springer.com/book/10.1007/978-3-642-54536-8).
- Check robot self-collision and collision between robot's bodies and objects.
- Plot robot kinematic chain and mesh using *matplotlib* or *trimesh* library

## Installation

### Requirements

You need a [python-fcl](https://github.com/BerkeleyAutomation/python-fcl) package to do object collision checking.

- On Ubuntu, install two dependency libraries using `apt`

  `sudo apt install liboctomap-dev`

  `sudo apt install libfcl-dev`
- On Mac, download the two dependency libraries from git repository and build it.

  - octomap

    `git clone https://github.com/OctoMap/octomap.git`

    ~~~shell
    $ cd octomap
    $ mkdir build
    $ cd build
    $ cmake ..
    $ sudo make
    $ sudo make install
    ~~~
  - fcl

    `git clone https://github.com/flexible-collision-library/fcl.git`

    Since python-fcl uses version 0.5.0 of fcl, checkout with tag 0.5.0

    ~~~shell
    $ cd fcl
    $ git checkout 0.5.0
    $ mkdir build
    $ cd build
    $ cmake ..
    $ sudo make
    $ sudo make install
    ~~~

### Install pykin

**pykin** supports macOS and Linux on Python 3.

- Install from pip

  ~~~shell
  $ pip3 or pip install pykin
  ~~~

- Install from source **[recommend]**

  ~~~shell
  $ git clone https://github.com/jdj2261/pykin.git
  $ cd pykin
  $ python3 seup.py install or sudo python3 setup.py install
  ~~~

- pykin directory structure

  ~~~
  └── pykin
      ├── assets
      ├── collision
      ├── examples
      ├── geometry
      ├── kinematics
      ├── models
      ├── robots
      └── utils
  ~~~

## Quick Start

You can see various examples in examples directory

- Robot Info

  You can see 7 robot info.

  `baxter, sawyer, iiwa14, iiwa7, panda, ur5e, doosan`

  ~~~shell
  $ cd pykin/examples
  $ python robot_info.py $(robot_name)
  # baxter
  $ python robot_info.py baxter
  # saywer
  $ python robot_info.py sawyer
  ~~~

- Forward kinematics

  You can compute the forward kinematics as well as visualize the visual or collision geometry.

  ~~~shell
  $ cd pykin/examples/forward_kinematics
  $ python robot_fk_baxter_test.py
  ~~~

  |                            visual                            |                          collision                           |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="img/baxter_plot_visual.png" width="400" height="300"/> | <img src="img/baxter_plot_collision.png" width="400" height="300"/> |

- Inverse Kinematics

  You can compute the inverse kinematics using levenberg marquardt(LM) or newton raphson(NR) method

  ~~~shell
  $ cd pykin/examples/inverse_kinematics
  $ python robot_ik_baxter_test.py
  ~~~

- Sampling based Inverse Kinematics

  You can compute the inverse kinematics using geometric-aware bayesian optimization(GaBO) method
  
  For more detailed information, check [GaBO module](/pykin/utils/gabo/)
  
  ~~~shell
  $ cd pykin/examples/inverse_kinematics
  $ python robot_ik_gabo_test.py
  ~~~

- Collision check

  The below images show the collision result as well as visualize robot using trimesh.Scene class

  ~~~shell
  $ cd pykin/examples/trimesh_renders
  $ python sawyer_render.py
  ~~~

  |                        trimesh.Scene                         |                            Result                            |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="img/sawyer_mesh_collision.png" width="200" height="200"/> | <img src="img/sawyer_collision_result.png" width="600" height="200"/> |

## Visualization

You can see visualization using matplotlib library or trimesh.Scene class.

- Visualize `simple urdf` using `matplotlib`


  |                        ur5e                        |                        sawyer                        |                        iiwa14                        |                        panda                        |
  | :------------------------------------------------: | :--------------------------------------------------: | :--------------------------------------------------: | :-------------------------------------------------: |
  | <img src="img/ur5e.png" width="200" height="200"/> | <img src="img/sawyer.png" width="200" height="200"/> | <img src="img/iiwa14.png" width="200" height="200"/> | <img src="img/panda.png" width="200" height="200"/> |


- Visualize `visual geometry` using `matplotlib`


  |                           ur5e                            |                           sawyer                            |                           iiwa14                            |                           panda                            |
  | :-------------------------------------------------------: | :---------------------------------------------------------: | :---------------------------------------------------------: | :--------------------------------------------------------: |
  | <img src="img/ur5e_visual.png" width="200" height="200"/> | <img src="img/sawyer_visual.png" width="200" height="200"/> | <img src="img/iiwa14_visual.png" width="200" height="200"/> | <img src="img/panda_visual.png" width="200" height="200"/> |


- Visualize `collision geometry` using `matplotlib`


  |                             ur5e                             |                            sawyer                            |                            iiwa14                            |                            panda                             |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="img/ur5e_collision.png" width="200" height="200"/> | <img src="img/sawyer_collision.png" width="200" height="200"/> | <img src="img/iiwa14_collision.png" width="200" height="200"/> | <img src="img/panda_collision.png" width="200" height="200"/> |

- Visualize mesh about `visual/collision geometry` using `trimesh.Scene class`

  ![baxter](img/all_robots.png)
