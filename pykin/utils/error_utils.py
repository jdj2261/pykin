class NotFoundError(Exception):
    """
    class of Custom Exception

    Args:
        data (all types): input data
    """
    def __init__(self, data):
        self.data = data
    
    def __str__(self):
        return f"Not Found {self.data}, please check the name again"