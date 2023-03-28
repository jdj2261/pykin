import os, io, datetime
import trimesh
import numpy as np
from copy import deepcopy
from PIL import Image
from pykin.utils import transform_utils as t_utils
from pykin.utils.plot_utils import createDirectory
import h5py


np.seterr(divide="ignore", invalid="ignore")

pykin_path = os.path.abspath(__file__ + "/../../")


def get_mesh_path(mesh_path, robot_name):
    result_path = pykin_path + "/assets/urdf/" + robot_name + "/"
    result_path = result_path + mesh_path
    return result_path


def get_object_mesh(mesh_name, scale=[1.0, 1.0, 1.0]):
    file_path = pykin_path + "/assets/objects/meshes/"
    mesh: trimesh.Trimesh = trimesh.load(file_path + mesh_name)
    mesh.apply_scale(scale)
    return mesh

# for acronym
def get_object_mesh_acronym(filename:str, mesh_root_dir:str, scale=[1.0, 1.0, 1.0]):
    if filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname))
    mesh.apply_scale(mesh_scale)
    return mesh

def get_mesh_bounds(mesh, pose=np.eye(4)):
    copied_mesh = deepcopy(mesh)
    copied_mesh.apply_transform(pose)
    return copied_mesh.bounds


def normalize(vec):
    return vec / np.linalg.norm(vec)


def surface_sampling(mesh, n_samples=2, face_weight=None):
    vertices, face_ind = trimesh.sample.sample_surface(
        mesh, count=n_samples, face_weight=face_weight
    )
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

    Gram-Schmidt orthonormalization

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

    if np.any(rot_axis, 0):
        unit_rot_axis = rot_axis / np.linalg.norm(rot_axis)
        R = t_utils.get_matrix_from_axis_angle(unit_rot_axis, angle)
    else:
        R = t_utils.get_matrix_from_axis_angle(rot_axis, angle)

    return R


def save_image(
    scene: trimesh.Scene,
    resolution=(1024, 1024),
    title="result",
    save_dir_name="result_images",
):
    createDirectory(save_dir_name)
    scene_data = scene.save_image(resolution=resolution)
    img = Image.open(io.BytesIO(scene_data))
    file_name = (
        save_dir_name
        + "/"
        + title
        + "_{}.png".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    print(f"Save {file_name}")
    img.save(file_name + ".png")
