class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = {}

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors