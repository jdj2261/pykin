import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import trimesh

import pykin.utils.transform_utils as t_utils

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

def get_rotation_from_vectors(A, B):
    unit_A = A / np.linalg.norm(A)
    unit_B = B / np.linalg.norm(B)
    dot_product = np.dot(unit_A, unit_B)
    angle = np.arccos(dot_product)

    rot_axis = np.cross(unit_B, unit_A)
    R = t_utils.get_matrix_from_axis_angle(rot_axis, angle)

    return R