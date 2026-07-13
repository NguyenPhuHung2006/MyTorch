class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = {}
        self.needs_input_grad = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors