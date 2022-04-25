from enum import Enum, auto
from pykin.action.activity import ActivityBase

class ReleaseStatus(Enum):
    """
    Grasp Status Enum class
    """
    PRE_RELEASE = auto()
    RELEASE = auto()
    POST_RELEASE = auto()

class PlaceAction(ActivityBase):
    def __init__(self):
        pass