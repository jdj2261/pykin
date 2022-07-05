import sys, os
import trimesh
import numpy as np
from copy import deepcopy

pykin_path = os.path.abspath(os.path.dirname(__file__)+"/../../" )
sys.path.append(pykin_path)

def get_mesh_path(mesh_path, robot_name):
    result_path = pykin_path + "/asset/urdf/" + robot_name +"/"
    result_path = result_path + mesh_path
    return result_path

def get_object_mesh(mesh_name, scale=[1.0, 1.0, 1.0]):
    file_path = pykin_path + '/asset/objects/meshes/'
    mesh = trimesh.load(file_path + mesh_name)
    mesh.apply_scale(scale)
    return mesh

def get_mesh_bounds(mesh, pose=np.eye(4)):
    copied_mesh = deepcopy(mesh)
    copied_mesh.apply_transform(pose)
    return copied_mesh.bounds