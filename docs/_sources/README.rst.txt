pykin
=====
Python Interface for the Robot Kinematics Library

This library has been created simply by referring to
`ikpy <https://github.com/Phylliade/ikpy.git>`__.

Features
--------

-  Pure python library
-  Support only URDF file
-  Compute Forward, Inverse Kinematics and Jacobian
-  There are two ways to find the IK solution, referring to the
   `Introduction to Humanoid Robotics
   book <https://link.springer.com/book/10.1007/978-3-642-54536-8>`__.
-  Compute Collision checkinkg
-  Plot Robot Kinematic Chain and Robot Mesh (STL file)

Installation
------------

Requirements
~~~~~~~~~~~~

You need a
`python-fcl <https://github.com/BerkeleyAutomation/python-fcl>`__
package to do object collision checking.

-  For Ubuntu, using ``apt``

``sudo apt install liboctomap-dev``

``sudo apt install libfcl-dev`` - For Mac, First, Download the source
and build it.

-  octomap

   ``git clone https://github.com/OctoMap/octomap.git``

   ::

       $ cd octomap
       $ mkdir build
       $ cd build
       $ cmake ..
       $ make
       $ make install

-  fcl

   ``git clone https://github.com/flexible-collision-library/fcl.git``

   Since python-fcl uses version 0.5.0 of fcl, checkout with tag 0.5.0

   ::

       $ cd fcl
       $ git checkout 0.5.0
       $ mkdir build
       $ cd build
       $ cmake ..
       $ make
       $ make install

If the above installation is complete

::

    pip install python-fcl

Install Pykin
~~~~~~~~~~~~~

::

    pip install pykin

When git clone, use the --recurse-submodules option.

The download may take a long time due to the large urdf file size.

::

    git clone --recurse-submodules https://github.com/jdj2261/pykin.git

Quick Start
-----------

-  Robot Info

You can see 4 example robot information.

``baxter, iiwa14, panda, and sawyer``

::

    import sys

    from pykin.robot import Robot
    from pykin.utils import plot_utils as plt

    file_path = '../../asset/urdf/baxter/baxter.urdf'

    if len(sys.argv) > 1:
        robot_name = sys.argv[1]
        file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
    robot = Robot(file_path)

    fig, ax = plt.init_3d_figure("URDF")

    """
    Only baxter and sawyer robots can see collisions.
    It is not visible unless sphere, cylinder, and box are defined in collision/geometry tags in urdf.
    """
    # If visible_collision is True, visualize collision
    plt.plot_robot(robot, 
                    transformations=robot.transformations,
                    ax=ax, 
                    name=robot.robot_name,
                    visible_collision=True)
    ax.legend()
    plt.show_figure()

- Forward Kinematics


::

    from pykin.robot import Robot from
    pykin.kinematics.transform import Transform from pykin.utils.kin\_utils
    import ShellColors as sc

    # baxter\_example file\_path = '../asset/urdf/baxter/baxter.urdf' robot
    = Robot(file\_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

    # set input joints head\_thetas = [0.0] right\_arm\_thetas = [0, 0, 0,
    0, 0, 0, 0] left\_arm\_thetas = [0, 0, 0, 0, 0, 0, 0] thetas =
    head\_thetas + right\_arm\_thetas + left\_arm\_thetas

    # compute FK fk = robot.kin.forward\_kinematics(thetas) for link,
    transform in fk.items(): print(f"{sc.HEADER}{link}{sc.ENDC},
    {transform.rot}, {transform.pos}")

- Jacobian

::

    from pykin.kinematics import transform as tf from
    pykin.robot import Robot

    # import jacobian from pykin.kinematics import jacobian as jac

    file\_path = '../asset/urdf/baxter/baxter.urdf' robot =
    Robot(file\_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

    left\_arm\_thetas = [0, 0, 0, 0, 0, 0, 0]

    # Before compute Jacobian, you must set from start link to end link
    robot.set\_desired\_frame("base", "left\_wrist") fk =
    robot.kin.forward\_kinematics(left\_arm\_thetas)

    # If you want to get Jacobian, use calc\_jacobian function J =
    jac.calc\_jacobian(robot.desired\_frames, fk, len(left\_arm\_thetas))
    print(J)

    right\_arm\_thetas = [0, 0, 0, 0, 0, 0, 0]
    robot.set\_desired\_frame("base", "right\_wrist") fk =
    robot.kin.forward\_kinematics(right\_arm\_thetas) J =
    jac.calc\_jacobian(robot.desired\_frames, fk, len(right\_arm\_thetas))
    print(J)

- Inverse Kinematics

::

    import numpy as np from pykin.robot import Robot from
    pykin.kinematics.transform import Transform

    # baxter\_example file\_path = '../asset/urdf/baxter/baxter.urdf' robot
    = Robot(file\_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

    # set joints for targe pose right\_arm\_thetas = np.random.randn(7)

    # set init joints init\_right\_thetas = np.random.randn(7)

    # Before compute IK, you must set from start link to end link
    robot.set\_desired\_frame("base", "right\_wrist")

    # Compute FK for target pose target\_fk =
    robot.kin.forward\_kinematics(right\_arm\_thetas)

    # get target pose target\_r\_pose =
    np.hstack((target\_fk["right\_wrist"].pos,
    target\_fk["right\_wrist"].rot))

    # Compute IK Solution using LM(Levenberg-Marquardt) or
    NR(Newton-Raphson) method ik\_right\_result, \_ =
    robot.kin.inverse\_kinematics(init\_right\_thetas, target\_r\_pose,
    method="LM")

    # Compare error btween Target pose and IK pose result\_fk =
    robot.kin.forward\_kinematics(ik\_right\_result) error =
    robot.compute\_pose\_error(
    target\_fk["right\_wrist"].homogeneous\_matrix,
    result\_fk["right\_wrist"].homogeneous\_matrix) 
    print(error)
