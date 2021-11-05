class ContactData:
    """
    Data structure for holding information about a collision contact.

    Args:
        names (str): The names of the two objects in order.
        contact (fcl.Contact): The contact in question.
    """

    def __init__(self, names, contact):
        self.names = set(names)
        self._inds = {
            names[0]: contact.b1,
            names[1]: contact.b2
        }
        self._point = contact.pos
        self._depth = contact.penetration_depth

    def __repr__(self):
        return 'pykin.collision.collision_manager.{}()'.format(type(self).__name__)

    @property
    def point(self):
        """
        The 3D point of intersection for this contact.
        
        Returns:
            (3,) float: The intersection point.
        """
        return self._point

    @property
    def depth(self):
        """
        The penetration depth of the 3D point of intersection for this contact.
        
        Returns:
            float: The penetration depth.
        """
        return self._depth

    def index(self, name):
        """
        Returns the index of the face in contact for the mesh with
        the given name.

        Args:
            name (str): The name of the target object.

        Returns:
            int: The index of the face in collision
        """
        return self._inds[name]
