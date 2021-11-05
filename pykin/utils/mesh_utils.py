import sys, os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"/../../" )
sys.path.append(pykin_path)


def get_mesh_path(mesh_path, robot_name):
    result_path = pykin_path + "/asset/urdf/" + robot_name +"/"
    result_path = result_path + mesh_path
    return result_path