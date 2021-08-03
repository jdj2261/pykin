class RobotParser(object):
    
    def __init__(self, filepath=None):
        self.root = None
        self.tree = None
        self.filepath = filepath
        if filepath is not None:
            self.parse(filepath)

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

    def parse(self, filepath):
        pass

    
