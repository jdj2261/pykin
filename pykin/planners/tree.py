class Tree:
    """
    Tree class
    """
    def __init__(self):
        self.vertices = []
        self.edges = []

    def add_vertex(self, q_joints):
        """
        Add q_joints in vertices

        Args:
            q_joints(np.array): joint angles
        """
        self.vertices.append(q_joints)
        
    def add_edge(self, q_joints_idx):
        """
        Add vertex indexes(parent, child) in edges

        Args:
            q_joints_idx(int): index of joint angle
        """
        self.edges.append(q_joints_idx)
