import numpy as np
import time
import trimesh
from pykin.utils import plot_utils as p_utils

JOINT_TYPE_MAP = {'revolute'  : 'revolute',
                  'fixed'     : 'fixed',
                  'prismatic' : 'prismatic'}

LINK_TYPE_MAP = {'cylinder' : 'cylinder',
                 'sphere'   : 'sphere',
                 'box'      : 'box',
                 'mesh'     : 'mesh'}

LINK_TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']

class ShellColors:
    COLOR_NC = '\033[0m' # No Color
    COLOR_BLACK='\033[0;30m'
    COLOR_GRAY='\033[1;30m'
    COLOR_RED='\033[0;31m'
    COLOR_LIGHT_RED='\033[1;31m'
    COLOR_GREEN='\033[0;32m'
    COLOR_LIGHT_GREEN='\033[1;32m'
    COLOR_BROWN='\033[0;33m'
    COLOR_YELLOW='\033[1;33m'
    COLOR_BLUE='\033[0;34m'
    COLOR_LIGHT_BLUE='\033[1;34m'
    COLOR_PURPLE='\033[0;35m'
    COLOR_LIGHT_PURPLE='\033[1;35m'
    COLOR_CYAN='\033[0;36m'
    COLOR_LIGHT_CYAN='\033[1;36m'
    COLOR_LIGHT_GRAY='\033[0;37m'
    COLOR_WHITE='\033[1;37m'

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    MAGENTA = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def convert_thetas_to_dict(active_joint_names, thetas):
    """
    Check if any pair of objects in the manager collide with one another.
    
    Args:
        active_joint_names (list): actuated joint names
        thetas (sequence of float): If not dict, convert to dict ex. {joint names : thetas}
    
    Returns:
        thetas (dict): Dictionary of actuated joint angles
    """
    if not isinstance(thetas, dict):
        assert len(active_joint_names) == len(thetas
        ), f"""the number of robot joint's angle is {len(active_joint_names)},
                but the number of input joint's angle is {len(thetas)}"""
        thetas = dict((j, thetas[i]) for i, j in enumerate(active_joint_names))
    return thetas


def logging_time(original_fn):
    """
    Decorator to check time of function
    """
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"WorkingTime[{original_fn.__name__}]: {end_time-start_time:.4f} sec\n")
        return result
    return wrapper_fn

def convert_string_to_narray(str_input):
    """
    Args:
        str_input (str): string

    Returns:
        np.array: Returns string to np.array
    """
    return np.array([float(data) for data in str_input.split()])

def calc_pose_error(tar_pose, cur_pose, EPS):
    """
    Args:
        tar_pos (np.array): target pose
        cur_pos (np.array): current pose
        EPS (float): epsilon

    Returns:
        np.array: Returns pose error
    """

    pos_err = np.array([tar_pose[:3, -1] - cur_pose[:3, -1]])
    rot_err = np.dot(cur_pose[:3, :3].T, tar_pose[:3, :3])
    w_err = np.dot(cur_pose[:3, :3], rot_to_omega(rot_err, EPS))

    return np.vstack((pos_err.T, w_err))


def rot_to_omega(R, EPS):
    # referred p36
    el = np.array(
            [[R[2, 1] - R[1, 2]],
            [R[0, 2] - R[2, 0]],
            [R[1, 0] - R[0, 1]]]
    )
    norm_el = np.linalg.norm(el)
    if norm_el > EPS:
        w = np.dot(np.arctan2(norm_el, np.trace(R) - 1) / norm_el, el)
    elif (R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0):
        w = np.zeros((3, 1))
    else:
        w = np.dot(np.pi/2, np.array([[R[0, 0] + 1], [R[1, 1] + 1], [R[2, 2] + 1]]))
    return w


def limit_joints(joint_angles, lower, upper):
    """
    Set joint angle limit

    Args:
        joint_angles (sequence of float): joint angles
        lower (sequence of float): lower limit
        upper (sequence of float): upper limit

    Returns:
        joint_angles (sequence of float): Returns limited joint angle 
    """
    if lower is not None and upper is not None:
        for i in range(len(joint_angles)):
            if joint_angles[i] < lower[i]:
                joint_angles[i] = lower[i]
            if joint_angles[i] > upper[i]:
                joint_angles[i] = upper[i]
    return joint_angles


def apply_objects_to_scene(trimesh_scene=None, objs=None):
    if trimesh_scene is None:
        trimesh_scene = trimesh.Scene()

    for obj_name, obj_info in objs.items():
        info = obj_info
        color = np.array(info.color)

        if info.gtype == "mesh":
            mesh = info.gparam
            mesh.visual.face_colors = color
            trimesh_scene.add_geometry(mesh, transform=info.h_mat)

        if info.gtype == "box":
            box_mesh = trimesh.creation.box(extents=info.gparam)
            box_mesh.visual.face_colors = color
            trimesh_scene.add_geometry(box_mesh, transform=info.h_mat)

        if info.gtype == "cylinder":
            capsule_mesh = trimesh.creation.cylinder(height=info.gparam[0], radius=info.gparam[1])
            capsule_mesh.visual.face_colors = color
            trimesh_scene.add_geometry(capsule_mesh, transform=info.h_mat)

        if info.gtype == "sphere":
            sphere_mesh = trimesh.creation.icosphere(radius=info.gparam)
            sphere_mesh.visual.face_colors = color
            trimesh_scene.add_geometry(sphere_mesh, transform=info.h_mat)

    return trimesh_scene

def apply_gripper_to_scene(trimesh_scene=None, robot=None):
    if trimesh_scene is None:
        trimesh_scene = trimesh.Scene()

    for link, info in robot.gripper.info.items():
        mesh = info[2]
        h_mat= info[3]
        if info[1] == 'mesh':
            for idx, mesh in enumerate(info[2]):
                mesh_color = p_utils.get_mesh_color(robot, link, 'collision', idx=idx)
                if len(info) > 4 :
                    mesh_color = info[4]
                mesh.visual.face_colors = mesh_color
                trimesh_scene.add_geometry(mesh, transform=h_mat)
    return trimesh_scene

def apply_robot_to_scene(trimesh_scene=None, robot=None, geom="collision"):
    if trimesh_scene is None:
        trimesh_scene = trimesh.Scene()

    for link, info in robot.info[geom].items():
        mesh = info[2]
        h_mat = info[3]
        
        if info[1] == "mesh":
            for idx, mesh in enumerate(info[2]):
                mesh_color = p_utils.get_mesh_color(robot, link, geom, idx)
                if len(info) > 4:
                    mesh_color = info[4]
                mesh.visual.face_colors = mesh_color
                trimesh_scene.add_geometry(mesh, transform=h_mat)
    
        if info[1] == "box":
            for idx, param in enumerate(info[2]):
                box_mesh = trimesh.creation.box(extents=param)
                box_color = p_utils.get_mesh_color(robot, link, geom, idx)
                box_mesh.visual.face_colors = box_color
                trimesh_scene.add_geometry(box_mesh, transform=h_mat)

        if info[1] == "cylinder":
            for idx, param in enumerate(info[2]):
                capsule_mesh = trimesh.creation.cylinder(height=param[0], radius=param[1])
                capsule_color = p_utils.get_mesh_color(robot, link, geom, idx)
                capsule_mesh.visual.face_colors = capsule_color
                trimesh_scene.add_geometry(capsule_mesh, transform=h_mat)

        if info[1] == "sphere":
            for idx, param in enumerate(info[2]):
                sphere_mesh = trimesh.creation.icosphere(radius=param)
                sphere_color = p_utils.get_mesh_color(robot, link, geom, idx)
                sphere_mesh.visual.face_colors = sphere_color
                trimesh_scene.add_geometry(sphere_mesh, transform=h_mat)
    return trimesh_scene


def get_mesh_param(link_type):
    file_name = str(link_type.gparam.get('filename'))
    color = link_type.gparam.get('color')
    color = np.array([color for color in color.values()]).flatten()
    return (file_name, color)


def get_cylinder_param(link_type):
    radius = float(link_type.gparam.get('radius'))
    length = float(link_type.gparam.get('length'))
    return (radius, length)


def get_spehre_param(link_type):
    radius = float(link_type.gparam.get('radius'))
    return radius


def get_box_param(link_type):
    size = list(link_type.gparam.get('size'))
    return size
