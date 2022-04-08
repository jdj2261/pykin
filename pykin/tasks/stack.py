import numpy as np
from enum import Enum, auto
from collections import OrderedDict
from copy import deepcopy

from pykin.tasks.grasp import GraspManager
from pykin.utils.task_utils import normalize, surface_sampling, projection, get_rotation_from_vectors, get_relative_transform
from pykin.utils.transform_utils import get_pose_from_homogeneous
from pykin.utils.log_utils import create_logger

# TODO
class StackManager(GraspManager):
    def __init__(
        self
    ):
        pass