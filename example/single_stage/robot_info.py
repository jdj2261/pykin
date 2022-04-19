import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

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

robot.show_robot_info()