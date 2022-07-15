import sys
file_path = 'urdf/baxter/baxter.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = 'urdf/' + robot_name + '/' + robot_name + '.urdf'
    if robot_name == "doosan":
        file_path = 'urdf/' + robot_name + '/' + "doosan_with_robotiq140" + '.urdf'
if "baxter" in file_path:
    from pykin.robots.bimanual import Bimanual
    robot = Bimanual(file_path)
else:
    from pykin.robots.single_arm import SingleArm
    robot = SingleArm(file_path)

robot.show_robot_info()