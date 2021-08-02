class RobotParser(object):
    
    def __init__(self, filename=None):
        self.root = None
        self.tree = None
        self.filename = filename
        if filename is not None:
            self.parse(filename)

    @property
    def root(self):
        return self._root
    
    @root.setter
    def root(self, root):
        self._root = root
    
    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree

    def parse(self, filename):
        pass

    
