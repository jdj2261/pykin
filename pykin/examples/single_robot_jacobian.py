import numpy as np
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.kinematics import jacobian as jac


file_path = "urdf/doosan/doosan_with_robotiq140.urdf"
robot = SingleArm(
    file_path,
    Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]),
    has_gripper=True,
    gripper_name="robotiq140",
)
robot.setup_link_name("base_0", "right_hand")
thetas = np.zeros(6)
fk = robot.forward_kin(thetas)

J = jac.calc_jacobian(robot.desired_frames, fk, len(np.zeros(7)))

print(J)
