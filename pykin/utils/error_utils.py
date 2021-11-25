class NotFoundError(Exception):
    """
    Class of custom Exception about Not Found

    Args:
        data (all types): input data
    """
    def __init__(self, data):
        self.data = data
    
    def __str__(self):
        return f"Not Found {self.data}, please check the name again"


class CollisionError(Exception):
    """
    Class of custom Exception about Collision

    Args:
        data (all types): input data
    """
    def __init__(self, data):
        self.data = data
    
    def __str__(self):
        return f"Check the collision.. {self.data}, please check settings again"

class LimitJointError(Exception):
    """
    Class of custom Exception about Collision

    Args:
        data (all types): input data
    """
    def __init__(self, *data):
        self.data = data
    
    def __str__(self):
        return f"Check the joints.. {self.data}, please check current joints setting again"

class OriValueError(Exception):
    """
    Class of custom Exception about Orientation Value

    Args:
        data (all types): input data
    """
    def __init__(self, data):
        self.data = data
    
    def __str__(self):
        return "Expecting the shape of the orientation to be (3,), (3,3), or (4,), instead got:""{}".format(self.data)

