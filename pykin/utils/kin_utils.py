import numpy as np
import time

from pykin.kinematics.transform import Transform

JOINT_TYPE_MAP = {'revolute': 'revolute',
                  'fixed': 'fixed',
                  'prismatic': 'prismatic'}

LINK_TYPE_MAP = { 'cylinder' : 'cylinder',
                  'sphere'   : 'sphere',
                  'box'      : 'box',
                  'mesh'     : 'mesh'}

LINK_TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']

class ShellColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Baxter:
    left_e0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.107, 0.,    0.   ])
    left_w0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.088, 0.,    0.   ])
    right_e0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.107, 0.,    0.   ])
    right_w0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.088, 0.,    0.   ])

    @staticmethod
    def add_visual_link(link_transforms, f):
        if "left_lower_shoulder" in f.link.name:
            link_transforms["left_upper_elbow_visual"] = np.dot(link_transforms["left_lower_shoulder"],
                                                                        Baxter.left_e0_fixed_offset)
        if "left_lower_elbow" in f.link.name:
            link_transforms["left_upper_forearm_visual"] = np.dot(link_transforms["left_lower_elbow"],
                                                                        Baxter.left_w0_fixed_offset)
        if "right_lower_shoulder" in f.link.name:
            link_transforms["right_upper_elbow_visual"] = np.dot(link_transforms["right_lower_shoulder"],
                                                                        Baxter.right_e0_fixed_offset)
        if "right_lower_elbow" in f.link.name:
            link_transforms["right_upper_forearm_visual"] = np.dot(link_transforms["right_lower_elbow"], 
                                                                        Baxter.right_w0_fixed_offset)


def convert_thetas_to_dict(active_joint_names, thetas):
    if not isinstance(thetas, dict):
        assert len(active_joint_names) == len(thetas
        ), f"""the number of joints is {len(active_joint_names)}, 
                but the number of joint's angle is {len(thetas)}"""
        thetas = dict((j, thetas[i]) for i, j in enumerate(active_joint_names))        
    return thetas

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(
            f"WorkingTime[{original_fn.__name__}]: {end_time-start_time:.4f} sec\n")
        return result
    return wrapper_fn


def convert_transform(origin):
    if origin is None:
        return Transform()
    else:
        return Transform(rot=origin.rot, pos=origin.pos)


def convert_string_to_narray(str_input):
    if str_input is not None:
        return np.array([float(data) for data in str_input.split()])


def calc_pose_error(T_ref, T_cur, EPS):

    def rot_to_omega(R):
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
            w = np.dot(
                np.pi/2, np.array([[R[0, 0] + 1], [R[1, 1] + 1], [R[2, 2] + 1]]))
        return w

    pos_err = np.array([T_ref[:3, -1] - T_cur[:3, -1]])
    rot_err = np.dot(T_cur[:3, :3].T, T_ref[:3, :3])
    w_err = np.dot(T_cur[:3, :3], rot_to_omega(rot_err))

    return np.vstack((pos_err.T, w_err))


def limit_joints(cur_jnt, lower, upper):
    if lower is not None and upper is not None:
        for i in range(len(cur_jnt)):
            if cur_jnt[i] < lower[i]:
                cur_jnt[i] = lower[i]
            if cur_jnt[i] > upper[i]:
                cur_jnt[i] = upper[i]
    return cur_jnt


def get_robot_geom(link):
    name = None
    gtype = None
    gparam = None

    if link.collision.gtype == "cylinder":
        name = link.name
        gtype = link.collision.gtype
        gparam = get_cylinder_param(link)
    elif link.collision.gtype == "sphere":
        name = link.name
        gtype = link.collision.gtype
        gparam = get_spehre_param(link)
    elif link.collision.gtype == "box":
        name = link.name
        gtype = link.collision.gtype
        gparam = get_box_param(link)

    return name, gtype, gparam


def get_cylinder_param(link):
    radius = float(link.collision.gparam.get('radius'))
    length = float(link.collision.gparam.get('length'))
    return (radius, length)


def get_spehre_param(link):
    radius = float(link.collision.gparam.get('radius'))
    return radius


def get_box_param(link):
    size = list(link.collision.gparam.get('size'))
    return size
