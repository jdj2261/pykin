class DistanceData:
    """
    Data structure for holding information about a distance query.
    """

    def __init__(self, names, result):
        """
        Initialize a DistanceData.

        Parameters
        ----------
        names : list of str
          The names of the two objects in order.
        contact : fcl.DistanceResult
          The distance query result.
        """
        self.names = set(names)
        self._inds = {
            names[0]: result.b1,
            names[1]: result.b2
        }
        self._points = {
            names[0]: result.nearest_points[0],
            names[1]: result.nearest_points[1]
        }
        self._distance = result.min_distance

    @property
    def distance(self):
        """
        Returns the distance between the two objects.

        Returns
        -------
        distance : float
          The euclidean distance between the objects.
        """
        return self._distance

    def index(self, name):
        """
        Returns the index of the closest face for the mesh with
        the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        index : int
          The index of the face in collisoin.
        """
        return self._inds[name]

    def point(self, name):
        """
        The 3D point of closest distance on the mesh with the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        point : (3,) float
          The closest point.
        """
        return self._points[name]
