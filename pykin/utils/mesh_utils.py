import os
import trimesh
import numpy as np
from copy import deepcopy

from pykin.utils import transform_utils as t_utils

np.seterr(divide='ignore', invalid='ignore')

pykin_path = os.path.abspath(__file__ + "/../../")

def get_mesh_path(mesh_path, robot_name):
    result_path = pykin_path + "/asset/urdf/" + robot_name +"/"
    result_path = result_path + mesh_path
    return result_path


def get_object_mesh(mesh_name, scale=[1.0, 1.0, 1.0]):
    file_path = pykin_path + '/asset/objects/meshes/'
    mesh:trimesh.Trimesh = trimesh.load(file_path + mesh_name)
    mesh.apply_scale(scale)
    return mesh


def get_mesh_bounds(mesh, pose=np.eye(4)):
    copied_mesh = deepcopy(mesh)
    copied_mesh.apply_transform(pose)
    return copied_mesh.bounds


def normalize(vec):
    return vec / np.linalg.norm(vec)

def surface_sampling(mesh, n_samples=2, face_weight=None):
    vertices, face_ind = trimesh.sample.sample_surface(mesh, count=n_samples, face_weight=face_weight)
    normals = mesh.face_normals[face_ind]
    return vertices, face_ind, normals

def projection(v, u):
    return np.dot(v, u) / np.dot(u, u) * u

def get_absolute_transform(A, B):
    # TA = B
    # T = B * inv(A)
    return np.dot(B, np.linalg.inv(A))

def get_relative_transform(A, B):
    # AT = B
    # T = inv(A) * B
    return np.dot(np.linalg.inv(A), B)

def get_grasp_directions(line, n_trials):
    """
    Generate grasp dicrections

    Args:
        line (np.array): line from vectorA to vector B
        n_trials (int): parameter to obtain grasp poses by 360/n_trials angle around a pair of contact points

    Returns:
        normal_dir (float): grasp direction
    """
    norm_vector = normalize(line)
    e1, e2 = np.eye(3)[:2]
    v1 = e1 - projection(e1, norm_vector)
    v1 = normalize(v1)
    v2 = e2 - projection(e2, norm_vector) - projection(e2, v1)
    v2 = normalize(v2)

    for theta in np.linspace(0, np.pi, n_trials):
        normal_dir = np.cos(theta) * v1 + np.sin(theta) * v2
        yield normal_dir

def get_rotation_from_vectors(A, B):
    unit_A = A / np.linalg.norm(A)
    unit_B = B / np.linalg.norm(B)
    dot_product = np.dot(unit_A, unit_B)
    angle = np.arccos(dot_product)

    rot_axis = np.cross(unit_B, unit_A)
    R = t_utils.get_matrix_from_axis_angle(rot_axis, angle)

    return R